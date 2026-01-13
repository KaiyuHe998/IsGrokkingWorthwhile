import json
import logging
import math
import os
import builtins
import random
import shutil
import warnings
from typing import Any, Dict, List, Optional
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
import functools

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .Cycle_gpt2 import GPT2WithCycle, GPT2WithCycleConfig

import torch.multiprocessing as mp
from tqdm.auto import tqdm, trange
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from torch.optim import AdamW
from transformers.optimization import Adafactor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizerFast,
    MBartConfig,
    MBartForConditionalGeneration,
    MBartTokenizerFast,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    BertConfig,
    BertModel,
    BertTokenizerFast,
    CamembertConfig,
    CamembertModel,
    CamembertTokenizerFast,
    DistilBertConfig,
    DistilBertModel,
    DistilBertTokenizerFast,
    ElectraConfig,
    ElectraModel,
    ElectraTokenizerFast,
    EncoderDecoderConfig,
    EncoderDecoderModel,
    LongformerConfig,
    LongformerModel,
    LongformerTokenizerFast,
    MarianConfig,
    MarianMTModel,
    MarianTokenizer,
    MobileBertConfig,
    MobileBertModel,
    MobileBertTokenizerFast,
    PreTrainedModel,
    PreTrainedTokenizerFast,
    RagTokenizer,
    RagRetriever,
    RagTokenForGeneration,
    RagSequenceForGeneration,
    RagConfig,
    RobertaConfig,
    RobertaModel,
    RobertaTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
)
import datasets
from datasets import load_from_disk

from simpletransformers.config.global_args import global_args
from simpletransformers.config.model_args import LanguageModelingArgs
from simpletransformers.config.utils import sweep_config_to_sweep_values
from simpletransformers.seq2seq.seq2seq_utils import (
    SimpleSummarizationDataset,
    load_hf_dataset,
)

try:
    import wandb

    wandb_available = True
except ImportError:
    wandb_available = False

if transformers.__version__ < "4.2.0":
    MBartForConditionalGeneration._keys_to_ignore_on_save = []

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "auto": (AutoConfig, AutoModel, AutoTokenizer),
    "bart": (BartConfig, BartForConditionalGeneration, BartTokenizerFast),
    "mbart": (MBartConfig, MBartForConditionalGeneration, MBartTokenizerFast),
    "mbart50": (MBartConfig, MBartForConditionalGeneration, MBart50TokenizerFast),
    "bert": (BertConfig, BertModel, BertTokenizerFast),
    "camembert": (CamembertConfig, CamembertModel, CamembertTokenizerFast),
    "distilbert": (DistilBertConfig, DistilBertModel, DistilBertTokenizerFast),
    "electra": (ElectraConfig, ElectraModel, ElectraTokenizerFast),
    "longformer": (LongformerConfig, LongformerModel, LongformerTokenizerFast),
    "mobilebert": (MobileBertConfig, MobileBertModel, MobileBertTokenizerFast),
    "marian": (MarianConfig, MarianMTModel, MarianTokenizer),
    "rag-token": (RagConfig, RagTokenForGeneration, RagTokenizer, RagRetriever),
    "rag-sequence": (RagConfig, RagSequenceForGeneration, RagTokenizer, RagRetriever),
    "roberta": (RobertaConfig, RobertaModel, RobertaTokenizerFast),
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "gpt2_cycle": (GPT2WithCycleConfig, GPT2WithCycle, GPT2Tokenizer),
}

class Seq2SeqModule(nn.Module):
    def __init__(
        self,
        language_model,
    ):
        super(Seq2SeqModule, self).__init__()
        self.lm = language_model

    def forward(self, target_ids=None, lm_labels=None):
        return self.lm(target_ids, labels=lm_labels)[0]

    def save_pretrained(self, output_dir):
        self.lm.save_pretrained(output_dir)
        self.lm.config.save_pretrained(output_dir)

    def generate(
        self,
        decoder_input_ids=None,              # should pad from left
        decoder_attention_mask=None,
        **kwargs,
    ):
        return self.lm.generate(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            **kwargs
        )
        

class Seq2SeqModel:
    def __init__(
        self,
        # lm args
        model_type,
        model_name,
        args=None,
        # ddp args
        ddp_args=None,
        use_cuda=True,
        new_tokens=None,
        init_weights=False,
        no_dropout=False,
        no_ln=False,
        no_mlp=False,
        share_mlp=False,
        add_memory=False,
        add_recurrence=False,
        re_embed=False,
        re_embed_temp=None,
        cuda_device=-1,
        relation_mean_shift=False,
        evaluate_train=False,
        **kwargs,
    ):

        print("numpy version:", np.__version__)
        print("torch version:", torch.__version__)
        print("transformers version:", transformers.__version__)
        ### load & update all general args
        self.args = self._load_model_args(model_name)
        self.args.evaluate_train = evaluate_train
        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, LanguageModelingArgs):
            self.args = args
        if "sweep_config" in kwargs:
            self.is_sweeping = True
            sweep_config = kwargs.pop("sweep_config")
            sweep_values = sweep_config_to_sweep_values(sweep_config)
            self.args.update_from_dict(sweep_values)
        else:
            self.is_sweeping = False
        ### ----------------

        ### GPU & distributed training setup
        self.local_rank = ddp_args["local_rank"]
        self.rank = ddp_args["rank"]
        self.gpu = ddp_args["gpu"]
        self.world_size = ddp_args["world_size"]
        self.dist_url = ddp_args["dist_url"]
        self.dist_backend = ddp_args["dist_backend"]

        self.args.n_gpu = torch.cuda.device_count()
        print("local gpu count:", self.args.n_gpu)
        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
        self.distributed = self.world_size > 1

        if self.distributed:
            print("***In distributed mode, world_size:{}***".format(self.world_size))

        if self.distributed:
            if self.local_rank != -1:  # for torch.distributed.launch
                print("provided local_rank is {}. Setting rank and gpu both to be the same.".format(self.local_rank))
                self.rank = self.local_rank
                self.gpu = self.local_rank
            elif 'SLURM_PROCID' in os.environ:  # for slurm scheduler
                self.rank = int(os.environ['SLURM_PROCID'])
                self.gpu = self.rank % self.args.n_gpu
                print("provided local_rank is -1. Setting rank and gpu with SLURM_PROCID. Rank:{}, gpu:{}"
                      .format(self.rank, self.gpu))
            dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url, world_size=self.world_size, rank=self.rank)
            assert self.rank >= 0
        else:
            assert self.rank == -1

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        if self.distributed:
            assert use_cuda

        if use_cuda:
            if torch.cuda.is_available():
                if self.local_rank == -1:
                    self.device = torch.device('cuda')
                else:
                    self.device = torch.device('cuda', self.local_rank)
            else:
                raise ValueError(
                    "'use_cuda' set to True when cuda is unavailable."
                    "Make sure CUDA is available or set `use_cuda=False`."
                )
        else:
            self.device = "cpu"
        print("setting device complete. device:", self.device)

        if not use_cuda:
            self.args.fp16 = False
        ### -----------
        
        ### load model and tokenizer
        _config_class, _model_class, _tokenizer_class = MODEL_CLASSES[model_type]
        if no_dropout:
            self.language_model = _model_class.from_pretrained(model_name, attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0, summary_first_dropout=0.0)
        else:
            self.language_model = _model_class.from_pretrained(model_name)
        self.lm_tokenizer = _tokenizer_class.from_pretrained(model_name)
        
        # set pad to be eos
        self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        self.lm_tokenizer.pad_token_id = self.lm_tokenizer.eos_token_id

        # add new tokens
        if new_tokens:
            self.lm_tokenizer.add_tokens(new_tokens)

        self.relation_mean_shift = relation_mean_shift
        if self.relation_mean_shift:
            # prepare the ID and OOD relation ids
            tokens = [self.lm_tokenizer.decode([i]) for i in range(len(self.lm_tokenizer))]
            ID_relation_ids = []
            for i in range(10000):
                rel = "<e_{}>".format(i)   # TODO
                if rel not in tokens:
                    continue
                assert tokens.count(rel) == 1
                ind = (self.lm_tokenizer.encode(rel)[0])
                ID_relation_ids.append(ind)

            OOD_relation_ids = []
            for i in range(10000):
                rel = "<n_e_{}>".format(i)
                if rel not in tokens:
                    continue
                assert tokens.count(rel) == 1
                ind = (self.lm_tokenizer.encode(rel)[0])
                OOD_relation_ids.append(ind)
            print("***ID/OOD relation mean shift***", "# ID/OOD relations:", len(ID_relation_ids), len(OOD_relation_ids))
            self.ID_relation_ids = ID_relation_ids
            self.OOD_relation_ids = OOD_relation_ids

        if self.args.wandb_project and not wandb_available:
            warnings.warn(
                "wandb_project specified but wandb is not available. Wandb disabled."
            )
            self.args.wandb_project = None

        if no_ln or no_mlp or share_mlp or add_memory:
            assert init_weights
        
        if init_weights:
            print("...initing weights...")
            temp_config = self.language_model.config
            temp_config.no_ln = no_ln
            temp_config.no_mlp = no_mlp
            temp_config.share_mlp = share_mlp
            if add_memory:
                print("adding memory; dimension:", self.args.memory_dim)
                temp_config.add_memory = add_memory
                temp_config.memory_dim = self.args.memory_dim
            temp_config.vocab_size = len(self.lm_tokenizer)
            if self.args.n_layer:
                temp_config.n_layer = self.args.n_layer
            if self.args.n_inner:
                temp_config.n_inner = self.args.n_inner
            if self.args.n_head:
                temp_config.n_head = self.args.n_head
            
            if model_type == "gpt2_cycle":
                if hasattr(self.args, "mlp_t"):
                    temp_config.mlp_t = self.args.mlp_t
                    print('Using MLP_t = True')
                if hasattr(self.args, "H_cycle"):
                    temp_config.num_cycles = self.args.H_cycle
                if hasattr(self.args, "L_cycle"):
                    temp_config.num_L_cycles = self.args.L_cycle
                if hasattr(self.args, "n_layer") and self.args.n_layer:
                    temp_config.num_layers = self.args.n_layer
                    temp_config.n_layer = self.args.n_layer
                if hasattr(self.args, "hidden_size") and self.args.hidden_size:
                    temp_config.n_embd = self.args.hidden_size
                    temp_config.n_head = self.args.hidden_size // 64
                    temp_config.n_inner = self.args.hidden_size * 4

            self.language_model = _model_class(temp_config)
            self.language_model.config = temp_config

        if add_recurrence:
            print("***in recurrence mode***")
            self.language_model.config.add_recurrence = add_recurrence ####

        if re_embed:
            assert add_recurrence
            print("***re-embedding***")
            self.language_model.config.re_embed = re_embed ####
            self.language_model.config.re_embed_temp = re_embed_temp

        # resize the embeddings, update config since new tokens are perhaps added
        self.language_model.resize_token_embeddings(len(self.lm_tokenizer), pad_to_multiple_of=8)
        self.language_model.config.vocab_size = self.language_model.get_input_embeddings().weight.shape[0]

        # if self.args.block_size <= 0:
        #     self.args.block_size = min(
        #         self.args.max_seq_length, self.lm_tokenizer.model_max_length
        #     )
        # else:
        #     self.args.block_size = min(
        #         self.args.block_size,
        #         self.lm_tokenizer.model_max_length,
        #         self.args.max_seq_length,
        #     )
        
        self.model = Seq2SeqModule(
            language_model=self.language_model,
        )

        self.args.model_type = model_type
        self.args.model_name = model_name

        print("### general model args:")
        print(self.args)
        print("### ddp args:")
        print(ddp_args)
        print("lm config:")
        print(self.language_model.config)


    def train_model(
        self,
        train_data,
        output_dir=None,
        show_running_loss=True,
        args=None,
        eval_data=None,
        test_data=None,
        verbose=True,
        save_step_dense=-1,
        save_step_dense_interval=-1,
        **kwargs,
    ):
        """
        Trains the model using 'train_data'

        Args:
            train_data: Pandas DataFrame containing the 2 columns - `input_text`, `target_text`.
                        - `input_text`: The input text sequence.
                        - `target_text`: The target text sequence
                        If `use_hf_datasets` is True, then this may also be the path to a TSV file with the same columns.
            output_dir: The directory where model files will be saved. If not given, self.args.output_dir will be used.
            show_running_loss (optional): Set to False to prevent running loss from being printed to console. Defaults to True.
            args (optional): Optional changes to the args dict of the model. Any changes made will persist for the model.
            eval_data (optional): A DataFrame against which evaluation will be performed when evaluate_during_training is enabled. Is required if evaluate_during_training is enabled.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down training significantly as the predicted sequences need to be generated.

        Returns:
            global_step: Number of global steps trained
            training_details: Average training loss if evaluate_during_training is False or full training progress scores if evaluate_during_training is True
        """  # noqa: ignore flake8"

        if args:
            self.args.update_from_dict(args)

        if self.distributed:
            self.args.silent = (self.rank != 0)

        if self.args.silent:
            show_running_loss = False

        if self.args.evaluate_during_training and eval_data is None:
            raise ValueError(
                "evaluate_during_training is enabled but eval_data is not specified."
                " Pass eval_data to model.train_model() if using evaluate_during_training."
            )

        if not output_dir:
            output_dir = self.args.output_dir

        if os.path.exists(output_dir) and os.listdir(output_dir) and not self.args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty."
                " Set args.overwrite_output_dir = True to overcome.".format(output_dir)
            )

        self._move_model_to_device()

        train_dataset = self.load_and_cache_examples(train_data, verbose=verbose)

        os.makedirs(output_dir, exist_ok=True)

        global_step, training_details = self.train(
            train_dataset,
            output_dir,
            show_running_loss=show_running_loss,
            eval_data=eval_data,
            test_data=test_data,
            verbose=verbose,
            save_step_dense=save_step_dense,
            save_step_dense_interval=save_step_dense_interval,
            **kwargs,
        )
        
        if verbose:
            logger.info(" Training of {} model complete. Saved to {}.".format(self.args.model_name, output_dir))

        return global_step, training_details

    def train(
        self,
        train_dataset,
        output_dir,
        show_running_loss=True,
        eval_data=None,
        test_data=None,
        verbose=True,
        save_step_dense=-1,
        save_step_dense_interval=-1,
        **kwargs,
    ):
        """
        Trains the model on train_dataset.

        Utility function to be used by the train_model() method. Not intended to be used directly.
        """

        model = self.model
        args = self.args
        lm_tokenizer = self.lm_tokenizer

        print("lm tokenizer:")
        print("\tbos:", lm_tokenizer.bos_token, lm_tokenizer.bos_token_id)
        print("\teos:", lm_tokenizer.eos_token, lm_tokenizer.eos_token_id)
        print("\tpad:", lm_tokenizer.pad_token, lm_tokenizer.pad_token_id)

        tb_writer = SummaryWriter(log_dir=args.tensorboard_dir)
        
        if self.distributed:
            print("invoking distributed sampler for rank", self.rank)
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.train_batch_size,
            num_workers=self.args.dataloader_num_workers,
        )

        if args.evaluate_during_training:
            eval_dataset = self.load_and_cache_examples(
                eval_data, verbose=verbose
            )
            if self.distributed:
                eval_sampler = DistributedSampler(eval_dataset, shuffle=True)
            else:
                eval_sampler = SequentialSampler(eval_dataset)

            eval_dataloader = DataLoader(
                eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
            )

        if args.max_steps > 0:
            t_total = args.max_steps
            args.num_train_epochs = (
                args.max_steps
                // (len(train_dataloader) // args.gradient_accumulation_steps)
                + 1
            )
        else:
            t_total = (
                len(train_dataloader)
                // args.gradient_accumulation_steps
                * args.num_train_epochs
            )

        no_decay = ["bias", "LayerNorm.weight", "ln"]          # params with no weight decay
        optimizer_grouped_parameters = []
        custom_parameter_names = set()
        for group in self.args.custom_parameter_groups:
            params = group.pop("params")
            custom_parameter_names.update(params)
            param_group = {**group}
            param_group["params"] = [
                p for n, p in model.named_parameters() if n in params
            ]
            optimizer_grouped_parameters.append(param_group)

        for group in self.args.custom_layer_parameters:
            layer_number = group.pop("layer")
            layer = f"layer.{layer_number}."
            group_d = {**group}
            group_nd = {**group}
            group_nd["weight_decay"] = 0.0
            params_d = []
            params_nd = []
            for n, p in model.named_parameters():
                if n not in custom_parameter_names and layer in n:
                    if any(nd in n for nd in no_decay):
                        params_nd.append(p)
                    else:
                        params_d.append(p)
                    custom_parameter_names.add(n)
            group_d["params"] = params_d
            group_nd["params"] = params_nd

            optimizer_grouped_parameters.append(group_d)
            optimizer_grouped_parameters.append(group_nd)

        if not self.args.train_custom_parameters_only:
            optimizer_grouped_parameters.extend(
                [
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and not any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in model.named_parameters()
                            if n not in custom_parameter_names
                            and any(nd in n for nd in no_decay)
                        ],
                        "weight_decay": 0.0,
                    },
                ]
            )

        for n, p in model.named_parameters():
            print(n, p.shape)
        
        num_total_params = 0
        print("# params:")
        for pg in optimizer_grouped_parameters:
            for p in pg['params']:
                temp = p.numel()
                print(temp, end="|")
                num_total_params += temp
        # print()
        print("total number of optimized params:", num_total_params)

        warmup_steps = math.ceil(t_total * args.warmup_ratio)
        args.warmup_steps = (
            warmup_steps if args.warmup_steps == 0 else args.warmup_steps
        )

        print("****************begin training. Total # of steps:", t_total, "warmup steps:", args.warmup_steps, "epochs:", args.num_train_epochs)

        if args.optimizer == "AdamW":
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
                betas=args.adam_betas,
            )
        elif args.optimizer == "Adafactor":
            optimizer = Adafactor(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adafactor_eps,
                clip_threshold=args.adafactor_clip_threshold,
                decay_rate=args.adafactor_decay_rate,
                beta1=args.adafactor_beta1,
                weight_decay=args.weight_decay,
                scale_parameter=args.adafactor_scale_parameter,
                relative_step=args.adafactor_relative_step,
                warmup_init=args.adafactor_warmup_init,
            )

        else:
            raise ValueError(
                "{} is not a valid optimizer class. Please use one of ('AdamW', 'Adafactor') instead.".format(
                    args.optimizer
                )
            )

        if args.scheduler == "constant_schedule":
            scheduler = get_constant_schedule(optimizer)

        elif args.scheduler == "constant_schedule_with_warmup":
            scheduler = get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=args.warmup_steps
            )

        elif args.scheduler == "linear_schedule_with_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
            )

        elif args.scheduler == "cosine_schedule_with_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "cosine_with_hard_restarts_schedule_with_warmup":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                num_cycles=args.cosine_schedule_num_cycles,
            )

        elif args.scheduler == "polynomial_decay_schedule_with_warmup":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=args.warmup_steps,
                num_training_steps=t_total,
                lr_end=args.polynomial_decay_schedule_lr_end,
                power=args.polynomial_decay_schedule_power,
            )

        else:
            raise ValueError("{} is not a valid scheduler.".format(args.scheduler))

        if (
            args.model_name
            and os.path.isfile(os.path.join(args.model_name, "optimizer.pt"))
            and os.path.isfile(os.path.join(args.model_name, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(args.model_name, "optimizer.pt"))
            )
            scheduler.load_state_dict(
                torch.load(os.path.join(args.model_name, "scheduler.pt"))
            )

        if self.distributed:
            # DDP
            if self.local_rank == -1:
                temp = 0
            else:
                temp = self.local_rank
            model = DDP(model, device_ids=[temp], output_device=temp)

        # in the distributed case, disable prints for non-master nodes
        if self.distributed:
            if self.rank != 0:
                print("I'm rank {}. I'm muted from now on.".format(self.rank))
                def print_pass(*args_):
                    pass
                builtins.print = print_pass
            else:
                print("I'm rank {}. I'll continue to print.".format(self.rank))


        logger.info(" Training started")

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        optimizer.zero_grad()
        train_iterator = trange(
            int(args.num_train_epochs), desc="Epoch", disable=args.silent, mininterval=0
        )
        epoch_number = 0
        steps_trained_in_current_epoch = 0
        epochs_trained = 0

        if args.model_name and os.path.exists(args.model_name):
            try:
                # set global_step to gobal_step of last saved checkpoint from model path
                checkpoint_suffix = args.model_name.split("/")[-1].split("-")
                if len(checkpoint_suffix) > 2:
                    checkpoint_suffix = checkpoint_suffix[1]
                else:
                    checkpoint_suffix = checkpoint_suffix[-1]
                global_step = int(checkpoint_suffix)
                epochs_trained = global_step // (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )
                steps_trained_in_current_epoch = global_step % (
                    len(train_dataloader) // args.gradient_accumulation_steps
                )

                logger.info(
                    "   Continuing training from checkpoint, will skip to saved global_step"
                )
                logger.info("   Continuing training from epoch %d", epochs_trained)
                logger.info("   Continuing training from global step %d", global_step)
                logger.info(
                    "   Will skip the first %d steps in the current epoch",
                    steps_trained_in_current_epoch,
                )
            except ValueError:
                logger.info("   Starting fine-tuning.")

        if args.evaluate_during_training:
            training_progress_scores = self._create_training_progress_scores(**kwargs)

        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                config={**asdict(args)},
                **args.wandb_kwargs,
            )
            wandb.run._label(repo="simpletransformers")
            wandb.watch(self.model)
            self.wandb_run_id = wandb.run.id

        if args.fp16:
            from torch.cuda import amp
            scaler = amp.GradScaler()

        # relation mean shift
        if self.relation_mean_shift:
            word_embedding = model.lm.lm_head.weight.data
            mean_ID = torch.mean(word_embedding[self.ID_relation_ids], dim=0)
            mean_OOD = torch.mean(word_embedding[self.OOD_relation_ids], dim=0)
            std_ID = torch.std(word_embedding[self.ID_relation_ids], dim=0)
            std_OOD = torch.std(word_embedding[self.OOD_relation_ids], dim=0)
            model.lm.lm_head.weight.data[self.OOD_relation_ids] = (word_embedding[self.OOD_relation_ids] - mean_OOD) / std_OOD * std_ID + mean_ID

        for current_epoch in train_iterator:

            current_epoch_losses = torch.zeros(3).to(self.device)
            steps_avg = 0

            model.train()

            if self.distributed:
                train_dataloader.sampler.set_epoch(current_epoch)

            if epochs_trained > 0:
                epochs_trained -= 1
                continue
            train_iterator.set_description(
                f"Epoch {epoch_number + 1} of {args.num_train_epochs}"
            )
            batch_iterator = tqdm(
                train_dataloader,
                desc=f"Running Epoch {epoch_number} of {args.num_train_epochs}",
                disable=args.silent,
                mininterval=0,
            )

            for step, batch in enumerate(batch_iterator):
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue

                inputs = self._get_inputs_dict(batch)
                # print(inputs['target_ids'][0])
                # print(inputs['lm_labels'][0])
                if args.fp16:
                    with amp.autocast():
                        loss = model(**inputs)
                else:
                    loss = model(**inputs)

                current_epoch_losses[0] += loss.item()
                steps_avg += 1

                if show_running_loss:
                    batch_iterator.set_description(
                        f"Epochs {epoch_number+1}/{args.num_train_epochs}. LM: {loss.item():9.4f}"  
                    )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                tr_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        scaler.unscale_(optimizer)
                    if args.optimizer == "AdamW":
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )

                    # for name, param in model.named_parameters():
                    #     if param.grad is None:
                    #         print("*********", name)

                    if args.fp16:
                        scale_before = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    # Update learning rate schedule
                    # Note: With AMP, scaler.step() may skip optimizer.step() on overflow.
                    if args.fp16:
                        scale_after = scaler.get_scale()
                        if scale_after < scale_before:
                            # Grad overflow: optimizer step was skipped, so also skip scheduler.
                            pass
                        else:
                            scheduler.step()
                    else:
                        scheduler.step()
                    model.zero_grad()
                    optimizer.zero_grad()

                    global_step += 1

                    # if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #     # Log metrics
                    #     tb_writer.add_scalar(
                    #         "lr", scheduler.get_last_lr()[0], global_step
                    #     )
                    #     tb_writer.add_scalar(
                    #         "loss",
                    #         (tr_loss - logging_loss) / args.logging_steps,
                    #         global_step,
                    #     )
                    #     logging_loss = tr_loss
                    #     if args.wandb_project or self.is_sweeping:
                    #         wandb.log(
                    #             {
                    #                 "Training loss": curr_LM_loss,
                    #                 "lr": scheduler.get_last_lr()[0],
                    #                 "global_step": global_step,
                    #             }
                    #         )

                    if ((args.save_steps > 0) and (global_step % args.save_steps == 0)) or (save_step_dense>0 and global_step % save_step_dense_interval == 0 and global_step<=save_step_dense):
                            # save/eval via step only when epoch number is less
                            output_dir_current = os.path.join(
                                output_dir, "checkpoint-{}".format(global_step)
                            )

                            self.save_model(
                                output_dir_current, optimizer, scheduler, model=model
                            )
                            
                            if args.evaluate_during_training:
                                results = self.eval_model(
                                    eval_dataloader,
                                    verbose=verbose,
                                    silent=args.evaluate_during_training_silent,
                                    **kwargs,
                                )
                                training_progress_scores["global_step"].append(global_step)
                                training_progress_scores["epoch"].append(-1)
                                training_progress_scores["train_loss"].append(-1.0)
                                for key in results:
                                    training_progress_scores[key].append(results[key])
                                report = pd.DataFrame(training_progress_scores)
                                report.to_csv(
                                    os.path.join(output_dir, "training_progress_scores.csv"),
                                    index=False,
                                )

                                if (not self.distributed) and args.predict_during_training:
                                    self.predict(test_data, output_dir_current, skip_model_moving=True)

                                model.train()

                    # relation mean shift
                    if self.relation_mean_shift:
                        word_embedding = model.lm.lm_head.weight.data
                        mean_ID = torch.mean(word_embedding[self.ID_relation_ids], dim=0)
                        mean_OOD = torch.mean(word_embedding[self.OOD_relation_ids], dim=0)
                        std_ID = torch.std(word_embedding[self.ID_relation_ids], dim=0)
                        std_OOD = torch.std(word_embedding[self.OOD_relation_ids], dim=0)
                        model.lm.lm_head.weight.data[self.OOD_relation_ids] = (word_embedding[self.OOD_relation_ids] - mean_OOD) / std_OOD * std_ID + mean_ID

                    
            current_epoch_losses[0] /= steps_avg
            current_epoch_losses[1] /= steps_avg
            current_epoch_losses[2] /= steps_avg
            if self.distributed:
                dist.all_reduce(current_epoch_losses, op=dist.ReduceOp.AVG)

            print("current_epoch_running_losses", current_epoch_losses)
            
            epoch_number += 1
            output_dir_current = os.path.join(
                output_dir, "checkpoint-{}-epoch-{}".format(global_step, epoch_number)
            )

            if args.save_model_every_epoch or (args.save_epoch_interval > 0 and epoch_number % args.save_epoch_interval == 0):
                os.makedirs(output_dir_current, exist_ok=True)
                self.save_model(output_dir_current, optimizer, scheduler, model=model)

                if args.evaluate_during_training:
                    results = self.eval_model(
                        eval_dataloader,
                        verbose=verbose,
                        silent=args.evaluate_during_training_silent,
                        **kwargs,
                    )

                    print(results)

                    training_progress_scores["global_step"].append(global_step)
                    training_progress_scores["epoch"].append(epoch_number)
                    training_progress_scores["train_loss"].append(current_epoch_losses[0].cpu().item())
                    for key in results:
                        training_progress_scores[key].append(results[key])
                    report = pd.DataFrame(training_progress_scores)
                    report.to_csv(
                        os.path.join(output_dir, "training_progress_scores.csv"),
                        index=False,
                    )

                    if (not self.distributed) and args.predict_during_training:
                        self.predict(test_data, output_dir_current, skip_model_moving=True)
            else:
                # if no saving via epoch, just record the training loss for the last epoch
                training_progress_scores["global_step"].append(global_step)
                training_progress_scores["epoch"].append(epoch_number)
                training_progress_scores["train_loss"].append(current_epoch_losses[0].cpu().item())
                training_progress_scores["eval_loss"].append(-1.0)
                report = pd.DataFrame(training_progress_scores)
                report.to_csv(
                    os.path.join(output_dir, "training_progress_scores.csv"),
                    index=False,
                )
            
        return (
            global_step,
            tr_loss / global_step
            if not self.args.evaluate_during_training
            else training_progress_scores,
        )

    def eval_model(
        self, eval_dataloader, verbose=True, silent=False, **kwargs
    ):
        """
        Evaluates the model on eval_data. Saves results to output_dir.

        Args:
            verbose: If verbose, results will be printed to the console on completion of evaluation.
            silent: If silent, tqdm progress bars will be hidden.
            **kwargs: Additional metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.
        Returns:
            results: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"

        # self._move_model_to_device()

        model = self.model
        args = self.args

        results = {}

        LM_loss = torch.zeros(1).to(self.device)
        nb_eval_steps = 0
        model.eval()

        # if args.n_gpu > 1:
        #     model = torch.nn.DataParallel(model)

        if self.args.fp16:
            from torch.cuda import amp

        for batch in tqdm(
            eval_dataloader, disable=args.silent or silent, desc="Running Evaluation"
        ):
            # batch = tuple(t.to(device) for t in batch)

            inputs = self._get_inputs_dict(batch)
            with torch.no_grad():
                if self.args.fp16:
                    with amp.autocast():
                        tmp_LM_loss = model(**inputs)
                else:
                    tmp_LM_loss = model(**inputs)
                # if self.args.n_gpu > 1:
                #     tmp_eval_loss = tmp_eval_loss.mean()
                LM_loss[0] += tmp_LM_loss.item()

            nb_eval_steps += 1

        LM_loss = LM_loss/nb_eval_steps
        if self.distributed:
            dist.all_reduce(LM_loss, op=dist.ReduceOp.AVG)

        results["eval_loss"] = LM_loss[0].cpu().item()

        return results


    def predict(
        self,
        pred_data,
        output_dir,
        cutoff=None,
        skip_model_moving=False,
        out_file="all_items.json",
        save_logit_lens_ood: bool = True,
        logit_lens_reasoning_pos_index: int = 3,
        logit_lens_topk: int = 50,
    ):
        """
        Performs generation.
        Params:
            pred_data: a list of items
            cutoff: if set, truncate the prediction set size
        """  # noqa: ignore flake8"

        all_items = []

        model = self.model.module if hasattr(self.model, "module") else self.model

        model.eval()
        # to_predict = pred_data["input_text"].tolist()
        # target_predict = pred_data["target_text"].tolist()
        to_predict = [item["input_text"] for item in pred_data]
        target_predict = [item["target_text"] for item in pred_data]

        if cutoff:
            to_predict = to_predict[:cutoff]
            target_predict = target_predict[:cutoff]

        if not skip_model_moving:
            self._move_model_to_device()

        self.lm_tokenizer.padding_side = "left" 
        self.lm_tokenizer.pad_token = self.lm_tokenizer.eos_token
        self.lm_tokenizer.pad_token_id = self.lm_tokenizer.eos_token_id

        def _id_to_token(token_id: int) -> str:
            try:
                # Prefer raw token pieces (keeps <e_*>/<r_*> intact)
                return self.lm_tokenizer.convert_ids_to_tokens(int(token_id))
            except Exception:
                return self.lm_tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)

        def _build_logit_lens_trace_for_batch(
            *,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            input_texts: List[str],
        ) -> List[List[Dict[str, Any]]]:
            """Logit-lens for *one-step* reasoning token (next-token) using per-layer hidden states.

            Returns: list (batch) of list (layers) of stage dicts.
            Each stage dict matches TRM-style keys: {stage, stream, predictions, ...}.
            """
            lm = model.lm  # underlying GPT2LMHeadModel / compatible
            with torch.no_grad():
                out = lm(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )

            hidden_states = out.hidden_states  # tuple: (emb, layer1, ..., layerN)
            if hidden_states is None:
                # Should not happen when output_hidden_states=True, but guard anyway.
                return [[] for _ in range(input_ids.size(0))]

            # Next-token logits come from hidden state at the last prompt position.
            # We take the last non-pad position per sample.
            # NOTE: padding_side is left, so attention_mask tells us valid tokens.
            last_pos = (attention_mask.long().sum(dim=1) - 1).clamp(min=0)  # [B]

            # Prepare fixed prefix tokens for TRM-style 5-slot predictions.
            # Input is typically <e_x><r_y><r_z> (3 tokens). We copy them into slots 0..2.
            # Slot 3 is the predicted next token (bridge-ish). Slot 4 is </a> (kept constant).
            prefix_tokens: List[List[str]] = []
            for txt in input_texts:
                toks = self.lm_tokenizer.tokenize(txt)
                # Keep only first 3 tokens for the classic inferred query format.
                # If input differs, still keep up to 3 for a stable layout.
                toks3 = toks[:3]
                while len(toks3) < 3:
                    toks3.append("<pad>")
                prefix_tokens.append(toks3)

            per_sample_traces: List[List[Dict[str, Any]]] = [[] for _ in range(input_ids.size(0))]
            lm_head = getattr(lm, "lm_head", None)
            if lm_head is None:
                raise AttributeError("Underlying LM has no lm_head; cannot build logit lens.")

            for layer_idx, hs in enumerate(hidden_states):
                # hs: [B, T, H]
                # gather last token hidden state per sample
                bsz = hs.size(0)
                last_h = hs[torch.arange(bsz, device=hs.device), last_pos]  # [B, H]
                logits = lm_head(last_h)  # [B, V]
                topk = min(int(logit_lens_topk), logits.size(-1))
                top_vals, top_ids = torch.topk(logits, k=topk, dim=-1)
                top1_ids = top_ids[:, 0]

                for bi in range(bsz):
                    top_tokens = [_id_to_token(tid) for tid in top_ids[bi].detach().cpu().tolist()]
                    top1_tok = _id_to_token(int(top1_ids[bi].detach().cpu().item()))

                    per_sample_traces[bi].append(
                        {
                            "stage": f"L{layer_idx}",
                            "stream": "decoder_hidden",
                            # TRM-style: 5 tokens; reasoning position is index=3
                            "predictions": [
                                prefix_tokens[bi][0],
                                prefix_tokens[bi][1],
                                prefix_tokens[bi][2],
                                top1_tok,
                                "</a>",
                            ],
                            # For downstream analysis that wants the full candidate set (within what we saved)
                            "positions": [
                                {
                                    "position": int(logit_lens_reasoning_pos_index),
                                    "predictions": top_tokens,
                                    "top_values": top_vals[bi].detach().cpu().float().tolist(),
                                }
                            ],
                            # lightweight hidden-state summary (avoid dumping full vectors)
                            "hidden_state_norm": float(torch.norm(last_h[bi]).detach().cpu().item()),
                        }
                    )

            return per_sample_traces

        all_outputs = []
        all_retrieved = []
        all_doc_scores = []
        # Batching
        for batch in tqdm(
            [
                to_predict[i : i + self.args.eval_batch_size]
                for i in range(0, len(to_predict), self.args.eval_batch_size)
            ],
            desc="Generating outputs",
            disable=self.args.silent,
        ):
            
            decoder_temp = self.lm_tokenizer(batch, return_tensors="pt", padding=True)
            decoder_input_ids, decoder_attention_mask = decoder_temp["input_ids"], decoder_temp["attention_mask"]
            
            decoder_input_ids, decoder_attention_mask = decoder_input_ids.to(self.device), decoder_attention_mask.to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                    max_length=self.args.max_length,
                    do_sample=False,
                )

            all_outputs.extend(outputs.cpu().numpy())
            
        if self.args.use_multiprocessed_decoding:
            if self.args.multiprocessing_chunksize == -1:
                chunksize = max(len(all_outputs) // (self.args.process_count * 2), 500)
            else:
                chunksize = self.args.multiprocessing_chunksize

            model.to("cpu")
            with Pool(self.args.process_count) as p:
                outputs = list(
                    tqdm(
                        p.imap(self._decode, all_outputs, chunksize=chunksize),
                        total=len(all_outputs),
                        desc="Decoding outputs",
                        disable=self.args.silent,
                    )
                )
            self._move_model_to_device()
        else:
            outputs = [
                self.lm_tokenizer.decode(
                    output_id,
                    skip_special_tokens=self.args.skip_special_tokens,
                    clean_up_tokenization_spaces=True,
                )
                for output_id in all_outputs
            ]

        # if self.args.num_return_sequences > 1:
        #     outputs = [
        #         outputs[i : i + self.args.num_return_sequences]
        #         for i in range(0, len(outputs), self.args.num_return_sequences)
        #     ]

        assert len(outputs) == len(to_predict)
        
        for i in range(len(to_predict)):
            outputs[i] = outputs[i].split("</a>")[0].strip()+"</a>"
            outputs[i] = "".join(outputs[i].split())
            # print("model output:\n\t", outputs[i])
            # print("target text:\n\t", target_predict[i])
            # print("---------------")
            all_items.append(pred_data[i])
            all_items[-1]["model_output"] = outputs[i] 

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, out_file), "w", encoding='utf-8') as f:
            json.dump(all_items, f)

        # Save logit lens for test_inferred_ood (TRM-compatible schema)
        if save_logit_lens_ood:
            try:
                ood_items = [it for it in all_items if isinstance(it, dict) and it.get("type") == "test_inferred_ood"]
                if len(ood_items) > 0:
                    ll_out_path = os.path.join(output_dir, "logit_lens_ood.json")
                    logit_lens_rows: List[Dict[str, Any]] = []

                    # Batch over OOD items for efficiency
                    ood_inputs = [it.get("input_text", "") for it in ood_items]
                    ood_targets = [it.get("target_text", "") for it in ood_items]
                    ood_model_outputs = [it.get("model_output", "") for it in ood_items]

                    for j in range(0, len(ood_inputs), self.args.eval_batch_size):
                        batch_texts = ood_inputs[j : j + self.args.eval_batch_size]
                        batch_targets = ood_targets[j : j + self.args.eval_batch_size]
                        batch_outputs = ood_model_outputs[j : j + self.args.eval_batch_size]

                        enc = self.lm_tokenizer(batch_texts, return_tensors="pt", padding=True)
                        in_ids = enc["input_ids"].to(self.device)
                        attn = enc["attention_mask"].to(self.device)

                        traces = _build_logit_lens_trace_for_batch(
                            input_ids=in_ids,
                            attention_mask=attn,
                            input_texts=batch_texts,
                        )

                        for bi in range(len(batch_texts)):
                            logit_lens_rows.append(
                                {
                                    "type": "test_inferred_ood",
                                    "input_text": batch_texts[bi],
                                    "target_text": batch_targets[bi],
                                    "model_final_output": batch_outputs[bi],
                                    "logit_lens_trace": traces[bi],
                                }
                            )

                    with open(ll_out_path, "w", encoding="utf-8") as f:
                        json.dump(logit_lens_rows, f)
            except Exception as e:
                # Don't break training/eval loops if logit lens fails.
                if not self.args.silent:
                    print(f"[WARN] Failed to save logit_lens_ood.json: {e}")

        self.lm_tokenizer.padding_side = "right"
        

    def _decode(self, output_id):
        return self.decoder_tokenizer.decode(
            output_id,
            skip_special_tokens=self.args.skip_special_tokens,
            clean_up_tokenization_spaces=True,
        )

    def compute_metrics(self, labels, preds, **kwargs):
        """
        Computes the evaluation metrics for the model predictions.

        Args:
            labels: List of target sequences
            preds: List of model generated outputs
            **kwargs: Custom metrics that should be used. Pass in the metrics as keyword arguments (name of metric: function to use).
                        A metric function should take in two parameters. The first parameter will be the true labels, and the second parameter will be the predictions. Both inputs
                        will be lists of strings. Note that this will slow down evaluation significantly as the predicted sequences need to be generated.

        Returns:
            result: Dictionary containing evaluation results.
        """  # noqa: ignore flake8"
        # assert len(labels) == len(preds)

        results = {}
        for metric, func in kwargs.items():
            results[metric] = func(labels, preds)

        return results

    def load_and_cache_examples(
        self, data, evaluate=False, no_cache=False, verbose=True, silent=False
    ):
        """
        Creates a Seq2SeqDataset from data.

        Utility function for train() and eval() methods. Not intended to be used directly.
        """

        lm_tokenizer = self.lm_tokenizer
        args = self.args

        if not no_cache:
            no_cache = args.no_cache

        if not no_cache:
            os.makedirs(self.args.cache_dir, exist_ok=True)

        mode = "dev" if evaluate else "train"
        
        #TODO: load from cache
        return SimpleSummarizationDataset(lm_tokenizer, self.args, data, mode)

    def _create_training_progress_scores(self, **kwargs):
        extra_metrics = {key: [] for key in kwargs}
        training_progress_scores = {
            "global_step": [],
            "epoch": [],
            "eval_loss": [],
            "train_loss": [],
            **extra_metrics,
        }

        return training_progress_scores

    def _get_last_metrics(self, metric_values):
        return {metric: values[-1] for metric, values in metric_values.items()}

    def save_model(
        self,
        output_dir=None,
        optimizer=None,
        scheduler=None,
        model=None,
        results=None,
    ):
        
        if self.distributed and self.rank != 0:
            # no saving for non-master nodes
            return

        if not output_dir:
            output_dir = self.args.output_dir
        
        # 即使不保存模型，也要创建目录，因为后续的 predict 需要写入 all_items.json
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Saving model into {output_dir}")

        # 🟢 新增逻辑：检查 evaluate_train 模式
        save_weights = True
        if getattr(self.args, "evaluate_train", False):
            try:
                # output_dir 通常格式为 ".../checkpoint-2000" 或 ".../checkpoint-2000-epoch-1"
                dir_name = os.path.basename(output_dir)
                if "checkpoint-" in dir_name:
                    # 提取 step 数字
                    step_str = dir_name.split("checkpoint-")[1].split("-")[0]
                    step = int(step_str)
                    # 如果 step > 2000，则不保存权重
                    if step > 2000:
                        save_weights = False
                        logger.info(f"evaluate_train=True: Skipping model weight/tokenizer save for step {step}")
            except Exception as e:
                logger.warning(f"Could not parse step from {output_dir}, saving weights by default. Error: {e}")

        # 🟢 修改判定条件：增加 save_weights 检查
        if model and not self.args.no_save and save_weights:
            try:
                # Take care of distributed/parallel training
                model_to_save = model.module if hasattr(model, "module") else model

                self.save_model_args(output_dir)
                os.makedirs(os.path.join(output_dir), exist_ok=True)
                model_to_save.save_pretrained(output_dir)
                self.lm_tokenizer.save_pretrained(os.path.join(output_dir))

                torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                if optimizer and scheduler and self.args.save_optimizer_and_scheduler:
                    torch.save(
                        optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt")
                    )
                    torch.save(
                        scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt")
                    )
            except Exception as e:
                # Torch save failures are very often caused by disk-full/quota or flaky FS.
                try:
                    usage = shutil.disk_usage(output_dir)
                    free_gb = usage.free / (1024**3)
                    total_gb = usage.total / (1024**3)
                    used_gb = usage.used / (1024**3)
                    logger.error(
                        f"Checkpoint save failed in {output_dir}. Disk usage: used={used_gb:.2f}GB total={total_gb:.2f}GB free={free_gb:.2f}GB"
                    )
                except Exception:
                    pass
                raise

        if results:
            output_eval_file = os.path.join(output_dir, "eval_results.txt")
            with open(output_eval_file, "w") as writer:
                for key in sorted(results.keys()):
                    writer.write("{} = {}\n".format(key, str(results[key])))

    def _move_model_to_device(self):
        self.model.to(self.device)

    def _get_inputs_dict(self, batch):
        device = self.device
        target_ids, lm_labels = (
            batch["target_ids"],
            batch["lm_labels"],
        )
        inputs = {
            "target_ids": target_ids.to(device),
            "lm_labels": lm_labels.to(device),
        }
        return inputs

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = LanguageModelingArgs()
        args.load(input_dir)
        return args
