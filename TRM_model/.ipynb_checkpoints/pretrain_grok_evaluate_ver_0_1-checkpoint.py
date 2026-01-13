import time
import json
from typing import Optional, Any, Sequence, List
from dataclasses import dataclass
import os
import math
import yaml
import shutil
import copy
import types
from transformers import GPT2Tokenizer

from typing import Optional, Dict, Any, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig
from adam_atan2 import AdamATan2

from puzzle_dataset import PuzzleDataset, PuzzleDatasetConfig, PuzzleDatasetMetadata
from utils.functions import load_model_class, get_model_source_path
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from models.ema import EMAHelper

# ==========================================
# Logit Lens Hook (for inference-trace analysis)
# ==========================================
CAPTURED_STATES = []

def instrumented_inner_forward(self, carry, batch: Dict[str, torch.Tensor]) -> Tuple[Any, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Monkey-patched forward used to capture intermediate reasoning states.
    Applied temporarily during evaluation; does not affect training.
    """
    global CAPTURED_STATES
    CAPTURED_STATES = []  # Clear history
    
    seq_info = dict(cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None)
    input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

    # Internal recording helper
    def record_state(stage_name, z_h_tensor, z_l_tensor):
        CAPTURED_STATES.append({
            "stage": stage_name,
            "z_H": z_h_tensor.detach().cpu().float(),
            "z_L": z_l_tensor.detach().cpu().float()
        })

    # Initialize
    z_H, z_L = carry.z_H, carry.z_L
    record_state("Init", z_H, z_L)
    
    # Unroll loop logic
    total_H_cycles = self.config.H_cycles
    for _H_step in range(total_H_cycles):
        # L Cycle (Thinking)
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
            record_state(f"H{_H_step}_L{_L_step}_Thinking", z_H, z_L)
            
        # H Update (Commit)
        z_H = self.L_level(z_H, z_L, **seq_info)
        record_state(f"H{_H_step}_Commit", z_H, z_L)

    # Repackage return values
    inner_carry_cls = type(carry)
    new_carry = inner_carry_cls(z_H=z_H.detach(), z_L=z_L.detach())
    output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
    q_logits = self.q_head(z_H[:, 0]).to(torch.float32)
    
    return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


@dataclass
class GenerateConfig:
    """Generation config."""
    max_inference_steps: int = 16
    return_logits: bool = False
    return_hidden_states: bool = False
    return_steps: bool = True
    device: str = "cuda"
    verbose: bool = False


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')
    name: str
    loss: LossConfig


class EvaluatorConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # Evaluators
    evaluators: List[EvaluatorConfig] = []

    # Hyperparams
    global_batch_size: int
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None
    min_eval_interval: Optional[int] = 0 # when to start eval
    eval_save_outputs: List[str] = []

    ema: bool = False # use Exponential-Moving-Average
    ema_rate: float = 0.999 # EMA-rate
    freeze_weights: bool = False # If True, freeze weights and only learn the embeddings
    post_fix: Optional[str] = None
    max_inference_steps: int = 16
    causal:bool = False
    maintain_prefix:bool = True

@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, split: str, rank: int, world_size: int, **kwargs):
    dataset = PuzzleDataset(PuzzleDatasetConfig(
        seed=config.seed,
        dataset_paths=config.data_paths_test if len(config.data_paths_test)>0 and split=="test" else config.data_paths,
        rank=rank,
        num_replicas=world_size,
        **kwargs
    ), split=split)
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=1,
        prefetch_factor=8,
        pin_memory=True,
        persistent_workers=True
    )
    return dataloader, dataset.metadata


def create_model(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        num_puzzle_identifiers=train_metadata.num_puzzle_identifiers,
        causal=config.causal # Non-autoregressive
    )

    # Instantiate model with loss head
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **config.arch.loss.__pydantic_extra__)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

        # Load checkpoint
        if rank == 0:
            load_checkpoint(model, config)

        # Broadcast parameters from rank 0
        if world_size > 1:
            with torch.no_grad():
                for param in list(model.parameters()) + list(model.buffers()):
                    dist.broadcast(param, src=0)

    # Optimizers and lr
    if config.arch.puzzle_emb_ndim == 0:
        optimizers = [
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.lr
        ]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr
        ]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size
            ),
            AdamATan2(
                model.parameters(),
                lr=0,  # Needs to be set by scheduler
                weight_decay=config.weight_decay,
                betas=(config.beta1, config.beta2)
            )
        ]
        optimizer_lrs = [
            config.puzzle_emb_lr,
            config.lr
        ]

    return model, optimizers, optimizer_lrs

def mix_weights_direct(device, alpha, net, nets):
    sd = []
    for i in range(len(nets)):
        sd += [nets[i].state_dict()]
    sd_alpha = {}
    for k in sd[0].keys():
        comb_net = alpha[0]*sd[0][k].to(device)
        for i in range(1,len(nets)):
            comb_net += alpha[i]*sd[i][k].to(device)
        sd_alpha[k] =  comb_net
    net.load_state_dict(sd_alpha)
    return net

def cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, base_lr: float, num_warmup_steps: int, num_training_steps: int, min_ratio: float = 0.0, num_cycles: float = 0.5
):
    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(max(1, num_warmup_steps))

    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return base_lr * (min_ratio + max(0.0, (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))))


def init_train_state(config: PretrainConfig, train_metadata: PuzzleDatasetMetadata, rank: int, world_size: int):
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_groups * train_metadata.mean_puzzle_examples / config.global_batch_size)

    # Model
    model, optimizers, optimizer_lrs = create_model(config, train_metadata, rank=rank, world_size=world_size)

    return TrainState(
        step=0,
        total_steps=total_steps,

        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None
    )


def save_train_state(config: PretrainConfig, train_state: TrainState):
    # FIXME: Only saved model.
    # TODO add checkpoint save path
    if config.checkpoint_path is None:
        return
    
    
    os.makedirs(config.checkpoint_path, exist_ok=True)
    torch.save(train_state.model.state_dict(), os.path.join(config.checkpoint_path, f"step_{train_state.step}"))


def load_checkpoint(model: nn.Module, config: PretrainConfig):
    if config.load_checkpoint is not None:
        print(f"Loading checkpoint {config.load_checkpoint}")

        # Load state dict
        state_dict = torch.load(config.load_checkpoint, map_location="cuda")

        # Resize and reset puzzle emb if needed
        puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        if puzzle_emb_name in state_dict:
            puzzle_emb = state_dict[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
                # Re-initialize using mean
                state_dict[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
        model.load_state_dict(state_dict, assign=True)


def compute_lr(base_lr: float, config: PretrainConfig, train_state: TrainState):
    return cosine_schedule_with_warmup_lr_lambda(
        current_step=train_state.step,
        base_lr=base_lr,
        num_warmup_steps=round(config.lr_warmup_steps),
        num_training_steps=train_state.total_steps,
        min_ratio=config.lr_min_ratio
    )



def create_evaluators(config: PretrainConfig, eval_metadata: PuzzleDatasetMetadata) -> List[Any]:
    data_paths =config.data_paths_test if len(config.data_paths_test)>0 else config.data_paths
    # Initialize evaluators
    evaluators = []
    for cfg in config.evaluators:
        for data_path in data_paths:
            cls = load_model_class(cfg.name, "evaluators.")(
                data_path=data_path, eval_metadata=eval_metadata, **cfg.__pydantic_extra__
            )  # type: ignore
            evaluators.append(cls)

    return evaluators

def train_batch(config: PretrainConfig, train_state: TrainState, batch: Any, global_batch_size: int, rank: int, world_size: int):
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return

    # To device
    batch = {k: v.cuda() for k, v in batch.items()}

    # Init carry if it is None
    if train_state.carry is None:
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

    # Forward
    train_state.carry, loss, metrics, _, _ = train_state.model(carry=train_state.carry, batch=batch, return_keys=[])

    ((1 / global_batch_size) * loss).backward()

    # Allreduce
    if world_size > 1:
        for param in train_state.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad)
            
    # Apply optimizer
    lr_this_step = None    
    for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
        lr_this_step = compute_lr(base_lr, config, train_state)

        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_step
            
        optim.step()
        optim.zero_grad()

    # Reduce metrics
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(sorted(metrics.keys()))  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if world_size > 1:
            dist.reduce(metric_values, dst=0)

        if rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}
            
            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {f"train/{k}": v / (global_batch_size if k.endswith("loss") else count) for k, v in reduced_metrics.items()}

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics
        
        
def load_tokenizer(config: PretrainConfig,):

    print("="*80)
    print("🔧 Load Tokenizer from checkpoints")
    print("="*80)

    # Select checkpoint path (can be changed to any checkpoint)
    assert len(config.data_paths) == 1, f'''Only one data folder should be passed in'''
    checkpoint_folder_name = config.data_paths[0].split('/')[-1]
    if 'finetuning' in checkpoint_folder_name:
        checkpoint_folder_name = checkpoint_folder_name.replace('_finetuning','')
    checkpoint_path = f'/home/kxh230002/GrokkedTransformer/output/{checkpoint_folder_name}/checkpoint-2000' # first tokenizer saved at the first checkpoint
    assert os.path.exists(checkpoint_path), f"Checkpoint does not exists: {checkpoint_path}"
    assert os.path.exists(os.path.join(checkpoint_path, "vocab.json")), f"Checkpoint lack vocab.json file: {checkpoint_path}, please run the grokking transformer main.py for initial run"
    print(f"\n📂 Checkpoint path: {checkpoint_path}")
    print(f"✓ check if path exists: {os.path.exists(checkpoint_path)}")

    # Method 1: load using transformers



    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint_path)

    
    # Tokenization sanity check
    test_text = "<e_0><r_1><a><e_2></a>"
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)


    # Method 2: load vocab.json manually (lower-level)


    vocab_path = os.path.join(checkpoint_path, "vocab.json")
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # Build token <-> id mappings
    token2id = vocab  # vocab.json is already in {token: id} format
    id2token = {v: k for k, v in token2id.items()}



    # Manual mapping sanity check
    test_tokens = ["<e_0>", "<r_1>", "<a>"]
    test_ids = [token2id.get(t, token2id.get("<unk>", 0)) for t in test_tokens]


    checkpoint_files = os.listdir(checkpoint_path)
    for f in sorted(checkpoint_files):
        file_path = os.path.join(checkpoint_path, f)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"   - {f:30s} ({size:>12,} bytes)")

    # Special token info

    special_tokens = ["<mask>", "<sep>", "<a>", "</a>", "<q>", "</q>"]
    for token in special_tokens:
        if token in token2id:
            print(f"   {token:10s} → ID: {token2id[token]}")

    
    return tokenizer


def generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    puzzle_identifier: Optional[int] = None,
    config: Optional[GenerateConfig] = None,
) -> Dict[str, Any]:
    """Generate predictions using the TRM model (with safety checks)."""
    model_config = model.model.inner.config
    if isinstance(model_config, dict):
        # Uncompiled model: config is a dict
        vocab_size = model_config['vocab_size']
        num_puzzle_identifiers = model_config['num_puzzle_identifiers']
    else:
        # Compiled model: config is a dataclass/object
        vocab_size = model_config.vocab_size
        num_puzzle_identifiers = model_config.num_puzzle_identifiers
    
    if config is None:
        config = GenerateConfig()
    
    model.eval()
    
    # 1) Prepare inputs
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    batch_size, seq_len = input_ids.shape
    assert batch_size == 1, "generate only support one input at a time"
    
    # Safety check: validate token id range
    max_token = input_ids.max().item()
    min_token = input_ids.min().item()
    
    if max_token >= vocab_size:
        raise ValueError(
            f"exceed token: {max_token} >= vocab_size ({vocab_size})\n"
            f"input sequence: {input_ids[0].tolist()}"
        )
    
    if min_token < 0:
        raise ValueError(
            f"negative token in inputs: {min_token}\n"
            f"input sequence: {input_ids[0].tolist()}"
        )
    
    if config.verbose:
        print(f"✓ Token check: [{min_token}, {max_token}] within [0, {vocab_size})")
    
    # Move to target device
    input_ids = input_ids.to(config.device)
    
    # Build batch dict
    if puzzle_identifier is None:
        puzzle_identifier = 0
    
    batch = {
        "inputs": input_ids,
        "labels": torch.full_like(input_ids, -100),
        "puzzle_identifiers": torch.tensor(
            [puzzle_identifier], 
            dtype=torch.long, 
            device=config.device
        )
    }
    
    if config.verbose:
        print(f"📥 Input shape: {input_ids.shape}")
        print(f"📥 Input tokens: {input_ids[0].tolist()}")
        print(f"📥 Puzzle ID: {puzzle_identifier}")
    
    # 2) Initialize carry
    with torch.device(config.device):
        carry = model.initial_carry(batch)
    
    if config.verbose:
        print(f"🔄 Initial carry created")
    
    # 3) Iterative inference
    return_keys = ["logits"]
    if config.return_hidden_states:
        return_keys.append("hidden_states")
    
    num_steps = 0
    all_hidden_states = [] if config.return_hidden_states else None
    
    with torch.inference_mode():
        for step in range(config.max_inference_steps):
            num_steps += 1
            
            try:
                carry, loss, metrics, preds, all_finish = model(
                    carry=carry,
                    batch=batch,
                    return_keys=return_keys,
                )
            except RuntimeError as e:
                if "index out of bounds" in str(e):
                    print(f"   vocab_size: {vocab_size}")
                    
                    # Try to inspect logits to see predicted tokens
                    if 'logits' in preds:
                        pred_tokens = torch.argmax(preds['logits'], dim=-1)
                raise
            
            if config.verbose and step < 3:
                print(f"   Step {step+1}: halted={carry.halted[0].item()}, "
                      f"steps={carry.steps[0].item():.2f}")
            
            if config.return_hidden_states and "hidden_states" in preds:
                all_hidden_states.append(preds["hidden_states"].cpu())
            
            if all_finish or carry.halted.all():
                if config.verbose:
                    print(f"✅ Stopped at step {num_steps}")
                break
    
    # 4) Extract results
    logits = preds["logits"][0]  # [L, vocab_size]
    pred_tokens = torch.argmax(logits, dim=-1)  # [L]
    
    # Sanity check: validate predicted token range
    pred_max = pred_tokens.max().item()
    if pred_max >= vocab_size:
        print(f"⚠️  warnning token {pred_max} >= vocab_size ({vocab_size})")
    
    # 5) Build return dict
    result = {
        "pred_tokens": pred_tokens.cpu(),
        "num_steps": num_steps,
        "halted": carry.halted[0].item(),
        "total_steps": carry.steps[0].item(),
    }
    
    if config.return_logits:
        result["logits"] = logits.cpu()
    
    if config.return_hidden_states and all_hidden_states:
        result["hidden_states"] = torch.stack(all_hidden_states, dim=0)
    
    if config.verbose:
        print(f"\n📤 Output:")
        print(f"   - Pred tokens: {pred_tokens.tolist()}")
        print(f"   - Num steps: {num_steps}")
        print(f"   - Total ACT steps: {carry.steps[0].item():.2f}")
    
    return result

def generate_batch(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    puzzle_identifiers: Optional[torch.Tensor] = None,
    config: Optional[GenerateConfig] = None,
) -> Dict[str, Any]:
    """Generate predictions in batch using the TRM model.

    Args:
        model: TRM model
        input_ids: input sequences [B, seq_len]
        puzzle_identifiers: puzzle IDs [B], optional
        config: generation config

    Returns:
        Dict containing:
            - pred_tokens: [B, seq_len] predicted token ids
            - num_steps: [B] number of inference steps per sample
            - halted: [B] whether each sample has halted
            - total_steps: [B] total ACT steps per sample
    """
    
    if config is None:
        config = GenerateConfig()
    
    model.eval()
    
    # Get model config
    model_config = model.model.inner.config
    if isinstance(model_config, dict):
        vocab_size = model_config['vocab_size']
        num_puzzle_identifiers = model_config['num_puzzle_identifiers']
    else:
        vocab_size = model_config.vocab_size
        num_puzzle_identifiers = model_config.num_puzzle_identifiers
    
    # 1) Prepare inputs
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    batch_size, seq_len = input_ids.shape
    
    # Safety check: validate token id range
    max_token = input_ids.max().item()
    min_token = input_ids.min().item()
    
    if max_token >= vocab_size:
        raise ValueError(
            f"token: {max_token} >= vocab_size ({vocab_size})"
        )
    
    if min_token < 0:
        raise ValueError(f"negative token: {min_token}")
    
    # Move to target device
    input_ids = input_ids.to(config.device)
    
    # Build puzzle_identifiers
    if puzzle_identifiers is None:
        puzzle_identifiers = torch.zeros(batch_size, dtype=torch.long, device=config.device)
    else:
        puzzle_identifiers = puzzle_identifiers.to(config.device)
    
    # Build batch dict
    batch = {
        "inputs": input_ids,
        "labels": torch.full_like(input_ids, -100),
        "puzzle_identifiers": puzzle_identifiers
    }
    
    # 2) Initialize carry
    with torch.device(config.device):
        carry = model.initial_carry(batch)
    
    # 3) Iterative inference
    return_keys = ["logits"]
    if config.return_hidden_states:
        return_keys.append("hidden_states")
    
    num_steps = torch.zeros(batch_size, dtype=torch.long, device=config.device)
    all_hidden_states = [] if config.return_hidden_states else None
    
    with torch.inference_mode():
        for step in range(config.max_inference_steps):
            # Update step counts for non-halted samples
            num_steps += (~carry.halted).long()
            
            carry, loss, metrics, preds, all_finish = model(
                carry=carry,
                batch=batch,
                return_keys=return_keys,
            )
            
            if config.return_hidden_states and "hidden_states" in preds:
                all_hidden_states.append(preds["hidden_states"].cpu())
            
            if all_finish or carry.halted.all():
                break
    
    # 4) Extract results
    logits = preds["logits"]  # [B, L, vocab_size]
    pred_tokens = torch.argmax(logits, dim=-1)  # [B, L]
    
    # 5) Build return dict
    result = {
        "pred_tokens": pred_tokens.cpu(),           # [B, L]
        "num_steps": num_steps.cpu(),               # [B]
        "halted": carry.halted.cpu(),               # [B]
        "total_steps": carry.steps.cpu(),           # [B]
    }
    
    if config.return_logits:
        result["logits"] = logits.cpu()             # [B, L, vocab_size]
    
    if config.return_hidden_states and all_hidden_states:
        result["hidden_states"] = torch.stack(all_hidden_states, dim=0)  # [T, B, L, D]
    
    return result


def tokenize_string(input_string,tokenizer,vocab_mapping, padding_len):
    tokenize_string = tokenizer.tokenize(input_string)
    
    print(f'tokenized input: {tokenize_string}' )
    token_ids = [vocab_mapping[i] for i in tokenize_string]
    token_ids.extend([0]*padding_len)
    ids = torch.tensor([token_ids]).to('cuda')
    return ids


def tokenize_batch(input_texts, tokenizer, vocab_mapping, seq_len=5):
    """
    Tokenize a batch of input strings.

    Args:
        input_texts: List[str] - list of input strings
        tokenizer: tokenizer instance
        vocab_mapping: token -> id mapping
        seq_len: sequence length (default: 5)

    Returns:
        torch.Tensor: token ids of shape [B, seq_len]
    """
    batch_token_ids = []
    
    for input_text in input_texts:
        tokens = tokenizer.tokenize(input_text)
        token_ids = [vocab_mapping[token] for token in tokens]
        
        # Padding
        if len(token_ids) < seq_len:
            token_ids.extend([0] * (seq_len - len(token_ids)))
        elif len(token_ids) > seq_len:
            token_ids = token_ids[:seq_len]
        
        batch_token_ids.append(token_ids)
    
    return torch.tensor(batch_token_ids, dtype=torch.long, device='cuda')

def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    rank: int,
    world_size: int,
    cpu_group: Optional[dist.ProcessGroup],
):
    # load_evaluate_data
    eval_model = train_state.model
    if hasattr(eval_model, '_orig_mod'):
        eval_model = eval_model._orig_mod
    all_items = []
    tokenizer = load_tokenizer(config)
    if config.checkpoint_path is not None:
        folder_name = os.path.basename(config.checkpoint_path)
        output_dir_name = f'TRM_{folder_name}'
    else:
        # Fallback
        key_parameters_str = f'HiddenDim_{config.arch.hidden_size}_MLP_{config.arch.mlp_t}_HaltMaxSteps_{config.arch.halt_max_steps}_Llayers_{config.arch.L_layers}_Lcycles_{config.arch.L_cycles}_Hlayers_{config.arch.H_layers}_Hcycles_{config.arch.H_cycles}_WeightDecay_{config.weight_decay}_LearningRate_{config.lr}'
        output_dir_name = f'TRM_{config.run_name}_{key_parameters_str}_{config.post_fix}'

    target_data_folder_name = config.data_paths[0].split('/')[-1]
    evaluate_data_path = f'../Grokking_analysis/data/{target_data_folder_name}/test.json'
    vocab_mapping_path = f'../Grokking_analysis/data/{target_data_folder_name}/vocab_map.json'
    reverse_vocab_mapping_path = f'../Grokking_analysis/data/{target_data_folder_name}/reverse_vocab_map.json'
    with open(reverse_vocab_mapping_path, 'r') as f:
        reverse_vocab_mapping = json.load(f)
    with open(vocab_mapping_path, 'r') as f:
        vocab_mapping = json.load(f)
    save_root_path = f'../Grokking_analysis/output/{output_dir_name}/checkpoint-{train_state.step}/all_items.json'
    
    if not os.path.exists(save_root_path):
        os.makedirs(os.path.dirname(save_root_path), exist_ok=True)

    with open(evaluate_data_path, 'r') as f:
        evaluate_data = json.load(f)
    eval_model.eval()
    seq_len = 5
        
    with torch.inference_mode():
        print('Now start generating model outputs for grokking evaluation...')
        from tqdm import tqdm
        
        batch_size = 512
        total_samples = len(evaluate_data)
        
        # Batch processing
        for batch_start in tqdm(range(0, total_samples, batch_size), desc="Evaluating batches"):
            batch_end = min(batch_start + batch_size, total_samples)
            batch_items = evaluate_data[batch_start:batch_end]
            
            # Batch extract input texts
            input_texts = [item['input_text'] for item in batch_items]
            
            # Batch tokenize
            batch_input_ids = tokenize_batch(input_texts, tokenizer, vocab_mapping, seq_len=seq_len)
            
            # Batch inference
            batch_results = generate_batch(
                model=eval_model,
                input_ids=batch_input_ids,
                config=GenerateConfig(
                    device='cuda',
                    max_inference_steps=config.max_inference_steps,
                    return_logits=False,
                    return_hidden_states=False,
                    verbose=False
                )
            )
            
            # Decode batch results
            batch_pred_tokens = batch_results['pred_tokens']  # [B, L]
            
            for idx, item in enumerate(batch_items):
                pred_tokens = batch_pred_tokens[idx].tolist()  # [L]
                
                # Remove PAD tokens (0)
                output_ids = [token_id for token_id in pred_tokens if token_id != 0]
                
                # Decode using reverse_vocab_mapping
                #output_tokens = [reverse_vocab_mapping[str(token_id)] for token_id in output_ids]
                output_tokens = [reverse_vocab_mapping.get(str(token_id), f"<UNK:{token_id}>") for token_id in output_ids]
                model_output = ''.join(output_tokens).replace('<PAD>', '')
                
                # Build result item
                tem_result = item.copy()
                tem_result.update({'model_output': model_output})
                all_items.append(tem_result)
    with open(save_root_path,'w') as f:
        json.dump(all_items,f,indent=4)
    
    if rank == 0 and config.checkpoint_path is not None:
        try:
            print("\n" + "="*80)
            print("🔍 Starting Logit Lens Analysis on OOD Test Set")
            print("="*80)
            
            # 1) Setup - get the uncompiled model
            logit_lens_model = train_state.model
            if hasattr(logit_lens_model, '_orig_mod'):
                logit_lens_model = logit_lens_model._orig_mod
            
            # Check model structure
            if not hasattr(logit_lens_model, 'model') or not hasattr(logit_lens_model.model, 'inner'):
                print("⚠️  Skip Logit Lens analysis")
            else:
                # Save original forward
                original_forward = logit_lens_model.model.inner.forward
                
                # Temporarily apply hook
                logit_lens_model.model.inner.forward = types.MethodType(
                    instrumented_inner_forward,
                    logit_lens_model.model.inner
                )
                
                print("🔧 Logit Lens Hook applied")
                
                # Filter OOD samples
                ood_samples = []
                for item in evaluate_data:
                    if item['type'] in ['test_inferred_ood', 
                                        'test_inferred_new',
                                        'test_inferred_new_mix',
                                        "test_inferred_new_id",   # hop1 NEW (ID), hop2 ID/OOD? (as you defined)
                                        "test_inferred_new_ood",
                                        "test_inferred_id_new",
                                        "test_inferred_ood_new",]:
                        ood_samples.append(item)

                print(f"   Found {len(ood_samples)} OOD samples for analysis")
                
                # 2) Inference + trace capture
                logit_lens_results = []
                lm_head = logit_lens_model.model.inner.lm_head
                puzzle_len = logit_lens_model.model.inner.puzzle_emb_len
                device = 'cuda'
                
                analysis_batch_size = 1  # Single-sample processing to ensure the hook captures correctly
                
                logit_lens_model.eval()
                with torch.inference_mode():
                    from tqdm import tqdm
                    for sample_idx in tqdm(range(len(ood_samples)), desc="Logit Lens Analysis"):  # Analyze all OOD samples
                        item = ood_samples[sample_idx]
                        input_text = item['input_text']
                        
                        # Tokenize
                        tokens = tokenizer.tokenize(input_text)
                        token_ids = [vocab_mapping.get(t, 0) for t in tokens]
                        
                        # Padding
                        if len(token_ids) < seq_len:
                            token_ids.extend([0] * (seq_len - len(token_ids)))
                        elif len(token_ids) > seq_len:
                            token_ids = token_ids[:seq_len]
                        
                        input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
                        
                        batch = {
                            "inputs": input_tensor,
                            "labels": torch.full_like(input_tensor, -100),
                            "puzzle_identifiers": torch.tensor([0], dtype=torch.long, device=device)
                        }
                        
                        # Run inference (triggers the hook)
                        with torch.device("cuda"):
                            carry = logit_lens_model.model.initial_carry(batch)
                        
                        # Forward (fills CAPTURED_STATES)
                        logit_lens_model.model(carry, batch)
                        
                        # 3) Process trace
                        trace_data = []
                        
                        for state in CAPTURED_STATES:
                            stage = state['stage']
                            
                            # Choose stream automatically
                            if "Thinking" in stage:
                                target_tensor = state['z_L']
                                stream_type = "z_L"
                            else:
                                target_tensor = state['z_H']
                                stream_type = "z_H"
                            
                            # Project to logits
                            target_tensor = target_tensor.to(device)
                            z_seq = target_tensor[:, puzzle_len:, :]
                            
                            with torch.no_grad():
                                logits = lm_head(z_seq)
                                top_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()
                            
                            # Decode tokens
                            decoded_tokens = [reverse_vocab_mapping.get(str(tid), f"<UNK:{tid}>") for tid in top_ids]
                            
                            trace_data.append({
                                "stage": stage,
                                "stream": stream_type,
                                "predictions": decoded_tokens
                            })
                        
                        # 4) Combine results
                        result_entry = {
                            "type": item.get("type", "unknown"),
                            "input_text": input_text,
                            "target_text": item.get("target_text"),
                            "model_final_output": trace_data[-1]["predictions"] if trace_data else [],
                            "logit_lens_trace": trace_data
                        }
                        
                        logit_lens_results.append(result_entry)
                
                # Restore original forward (important!)
                logit_lens_model.model.inner.forward = original_forward
                print("✅ recover to origional forward")
                
                # 5) Save results (same directory as all_items.json)
                logit_lens_save_dir = os.path.dirname(save_root_path)
                logit_lens_save_path = os.path.join(
                    logit_lens_save_dir,
                    "logit_lens_ood.json"
                )
                
                print(f"\n💾 Saving Logit Lens results to: {logit_lens_save_path}")
                os.makedirs(logit_lens_save_dir, exist_ok=True)
                with open(logit_lens_save_path, 'w', encoding='utf-8') as f:
                    json.dump(logit_lens_results, f, indent=2, ensure_ascii=False)
                
                print(f"✅ Logit Lens Analysis Complete! Analyzed {len(logit_lens_results)} samples.")
                print("="*80 + "\n")
        
        except Exception as e:
            print(f"⚠️  Logit Lens fail: {e}")
            import traceback
            traceback.print_exc()
    
    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0
        
        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")
            
            # To device
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(v.cpu())  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())), dtype=torch.float32, device="cuda"
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds, os.path.join(config.checkpoint_path, f"step_{train_state.step}_all_preds.{rank}")
            )

        del save_preds

        # Reduce to rank 0
        if metric_values is not None:
            if world_size > 1:
                dist.reduce(metric_values, dst=0)

            if rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")
            
        for i, evaluator in enumerate(evaluators):
            if rank == 0:
                print(f"Running evaluator {i+1}/{len(evaluators)}: {evaluator.__class__.__name__}")
                
            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(evaluator_save_path, rank=rank, world_size=world_size, group=cpu_group)
            if rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")
                
        if rank == 0:
            print("All evaluators completed!")

    return reduced_metrics

def save_code_and_config(config: PretrainConfig):
    if config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name),
        get_model_source_path(config.arch.loss.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    config_file = os.path.join(config.checkpoint_path, "all_config.yaml")
    with open(config_file, "wt") as f:
        yaml.dump(config.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def load_synced_config(hydra_config: DictConfig, rank: int, world_size: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**hydra_config)  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{os.path.basename(config.data_paths[0]).capitalize()}-ACT-torch"
        if config.run_name is None:
            config.run_name = f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}"
        if config.checkpoint_path is None:
            folder_name = config.run_name
            if hasattr(config, 'post_fix') and config.post_fix:
                key_parameters_str = f'HiddenDim_{config.arch.hidden_size}_MLP_{config.arch.mlp_t}_HaltMaxSteps_{config.arch.halt_max_steps}_Llayers_{config.arch.L_layers}_Lcycles_{config.arch.L_cycles}_Hlayers_{config.arch.H_layers}_Hcycles_{config.arch.H_cycles}_WeightDecay_{config.weight_decay}_LearningRate_{config.lr}'
                folder_name = f"{config.run_name}_{key_parameters_str}_{config.post_fix}"
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, folder_name)

        objects = [config]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    RANK = 0
    WORLD_SIZE = 1
    CPU_PROCESS_GROUP = None

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()

        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        
        # CPU GLOO process group
        CPU_PROCESS_GROUP = dist.new_group(backend="gloo")
        assert (
            dist.get_rank(CPU_PROCESS_GROUP) == RANK and dist.get_world_size(CPU_PROCESS_GROUP) == WORLD_SIZE
        )

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK, world_size=WORLD_SIZE)

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # Dataset
    train_epochs_per_iter = config.eval_interval if config.eval_interval is not None else config.epochs
    total_iters = config.epochs // train_epochs_per_iter

    assert config.epochs % train_epochs_per_iter == 0, "Eval interval must be a divisor of total epochs."

    train_loader, train_metadata = create_dataloader(config, "train", test_set_mode=False, epochs_per_iter=train_epochs_per_iter, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    try:
        eval_loader,  eval_metadata  = create_dataloader(config, "test", test_set_mode=True, epochs_per_iter=1, global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE)
    except:
        print("NO EVAL DATA FOUND")
        eval_loader = eval_metadata = None

    try:
        evaluators = create_evaluators(config, eval_metadata)
    except Exception as e:
        import traceback
        print("No evaluator found:", repr(e))
        traceback.print_exc()
        evaluators = []

    # Train state
    train_state = init_train_state(config, train_metadata, rank=RANK, world_size=WORLD_SIZE)

    # Progress bar and logger
    progress_bar = None
    ema_helper = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps)
        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump(), settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config)
    if config.ema:
        print('Setup EMA')
        ema_helper = EMAHelper(mu=config.ema_rate)
        ema_helper.register(train_state.model)

    # Training Loop
    for _iter_id in range(total_iters):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {_iter_id * train_epochs_per_iter}")

        ############ Train Iter
        if RANK == 0:
            print("TRAIN")
        train_state.model.train()
        for set_name, batch, global_batch_size in train_loader:
            metrics = train_batch(config, train_state, batch, global_batch_size, rank=RANK, world_size=WORLD_SIZE)

            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
            if config.ema:
                ema_helper.update(train_state.model)

        if _iter_id >= config.min_eval_interval:
            ############ Evaluation
            if RANK == 0:
                print("EVALUATE")
            if config.ema:
                print("SWITCH TO EMA")
                train_state_eval = copy.deepcopy(train_state)
                train_state_eval.model = ema_helper.ema_copy(train_state_eval.model)
            else:
                train_state_eval = train_state
            train_state_eval.model.eval()
            metrics = evaluate(config, 
                train_state_eval, 
                eval_loader, 
                eval_metadata, 
                evaluators,
                rank=RANK, 
                world_size=WORLD_SIZE,
                cpu_group=CPU_PROCESS_GROUP)


            if RANK == 0 and metrics is not None:
                wandb.log(metrics, step=train_state.step)
                
            ############ Checkpointing
            if RANK == 0:
                print("SAVE CHECKPOINT")
            if RANK == 0 and (config.checkpoint_every_eval or (_iter_id == total_iters - 1)):
                save_train_state(config, train_state_eval)

            if config.ema:
                del train_state_eval

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()


if __name__ == "__main__":
    launch()
