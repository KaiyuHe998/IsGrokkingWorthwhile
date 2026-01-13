import torch
import torch.nn as nn
from transformers import GPT2PreTrainedModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from typing import Optional, Tuple
from dataclasses import dataclass

class SwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion=4):
        super().__init__()
        intermediate_size = int(hidden_size * expansion)
        self.w1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w2 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w3 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.w3(self.silu(self.w1(x)) * self.w2(x))


class ReasoningBlock(nn.Module):
    def __init__(self, config, layer_idx=0):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        

        if not config.mlp_t:

            self.attn = GPT2Attention(config, layer_idx=layer_idx)
        else:

            self.attn = None 
            self.seq_mlp = SwiGLU(hidden_size=config.reasoning_seq_len, expansion=2)

        self.mlp = GPT2MLP(config.n_embd, config)

    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, **kwargs):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        if not self.config.mlp_t:
            # === Attention Mode ===
            attn_outputs = self.attn(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask,
                **kwargs
            )
            attn_output = attn_outputs[0]
            hidden_states = residual + attn_output
        else:
            # === MLP Router Mode ===
            # [B, L, D] -> Transpose -> [B, D, L]
            hidden_states_T = hidden_states.transpose(1, 2)
            

            curr_len = hidden_states_T.shape[-1]
            target_len = self.config.reasoning_seq_len
            

            if curr_len != target_len:
                hidden_states_T = torch.nn.functional.interpolate(hidden_states_T, size=target_len, mode='linear')
            
            out_T = self.seq_mlp(hidden_states_T)
            
            if curr_len != target_len:
                out_T = torch.nn.functional.interpolate(out_T, size=curr_len, mode='linear')

            out = out_T.transpose(1, 2)
            
            
            hidden_states = residual + out

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        return (hidden_states,)

# --------------------------
# 3. Config and Carry
# --------------------------
@dataclass
class CycleCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor
    
    def detach(self):
        return CycleCarry(z_H=self.z_H.detach(), z_L=self.z_L.detach())

class GPT2WithCycleConfig(GPT2Config):
    model_type = "gpt2_cycle"
    
    def __init__(
        self,
        num_cycles: int = 3,
        num_L_cycles: int = 1,
        num_layers: int = 2,
        train_last_cycle_only: bool = True,
        mlp_t: bool = True,           
        reasoning_seq_len: int = 64, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_cycles = num_cycles
        self.num_L_cycles = num_L_cycles
        self.num_layers = num_layers
        self.n_layer = num_layers 
        self.train_last_cycle_only = train_last_cycle_only
        self.mlp_t = mlp_t 
        self.reasoning_seq_len = reasoning_seq_len

class CycleReasoningModule(nn.Module):
    def __init__(self, config: GPT2WithCycleConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            ReasoningBlock(config, layer_idx=i) 
            for i in range(config.num_layers)
        ])
        self.ln = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, z_L, z_H, input_injection, attention_mask=None):
        # Step 1: Update L-level
        for _ in range(self.config.num_L_cycles):
            hidden_L = z_L + z_H + input_injection
            for layer in self.layers:
                outputs = layer(hidden_L, attention_mask=attention_mask)
                hidden_L = outputs[0]
            z_L = self.ln(hidden_L)
        
        # Step 2: Update H-level
        hidden_H = z_H + z_L
        for layer in self.layers:
            outputs = layer(hidden_H, attention_mask=attention_mask)
            hidden_H = outputs[0]
        z_H = self.ln(hidden_H)
        
        return z_L, z_H

# --------------------------
# 4. (GPT2WithCycle)
# --------------------------
class GPT2WithCycle(GPT2PreTrainedModel):
    config_class = GPT2WithCycleConfig
    
    def __init__(self, config: GPT2WithCycleConfig):
        super().__init__(config)
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        
        self.cycle_module = CycleReasoningModule(config)
        
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        
        self.initial_z_L = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)
        self.initial_z_H = nn.Parameter(torch.randn(1, 1, config.n_embd) * 0.02)
        
        self.post_init()
        self.lm_head.weight = self.wte.weight


    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    # ========================================================

    def get_initial_carry(self, batch_size, seq_len, device):
        return CycleCarry(
            z_L=self.initial_z_L.expand(batch_size, seq_len, -1).to(device),
            z_H=self.initial_z_H.expand(batch_size, seq_len, -1).to(device),
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        carry: Optional[CycleCarry] = None,
        return_carry: bool = False,
        **kwargs
    ):
        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        device = input_ids.device

        # Embedding
        inputs_embeds = self.wte(input_ids)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(0)
        position_embeds = self.wpe(position_ids)
        input_injection = self.drop(inputs_embeds + position_embeds)

        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0

        if carry is None:
            carry = self.get_initial_carry(batch_size, seq_len, device)
        z_L, z_H = carry.z_L, carry.z_H


        
        num_cycles = self.config.num_cycles

        if self.config.train_last_cycle_only and self.training:
            
            with torch.no_grad():
                for _ in range(num_cycles - 1):
                    z_L, z_H = self.cycle_module(z_L, z_H, input_injection, attention_mask)
                    z_L, z_H = z_L.detach(), z_H.detach()
            # Last cycle with grad
            z_L, z_H = self.cycle_module(z_L, z_H, input_injection, attention_mask)
        else:

            for _ in range(num_cycles):
                 z_L, z_H = self.cycle_module(z_L, z_H, input_injection, attention_mask)

        # =================================================================

        hidden_states = self.ln_f(z_H)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift logits for loss calculation
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        return {"input_ids": input_ids}