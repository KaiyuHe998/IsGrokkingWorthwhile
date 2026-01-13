"""
Decoder-Only Transformer 实现
标准的 GPT-style causal transformer，用于基线对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DecoderOnlyTransformer(nn.Module):
    """
    标准的 Decoder-Only Transformer (类似 GPT)
    用于和 TRM 模型进行基线对比
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.num_heads = config['num_heads']
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config['max_seq_len']
        self.dropout = config.get('dropout', 0.1)
        
        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        
        # Position embedding
        self.position_embedding = nn.Embedding(self.max_seq_len, self.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Layer norm
        self.ln_f = nn.LayerNorm(self.hidden_size)
        
        # Output projection
        self.lm_head = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        
        # Weight tying (可选)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len)
        
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # (B, T, H)
        
        # Position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, -1)  # (B, T)
        pos_embeds = self.position_embedding(positions)  # (B, T, H)
        
        # Combine embeddings
        x = token_embeds + pos_embeds
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (B, T, V)
        
        return logits
    
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None):
        """
        自回归生成
        
        Args:
            input_ids: (batch_size, seq_len) - prompt tokens
            max_new_tokens: 最多生成多少个新 token
            temperature: 采样温度
            top_k: top-k 采样
        
        Returns:
            generated: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 如果序列太长，截断
                idx_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
                
                # Forward
                logits = self(idx_cond)  # (B, T, V)
                
                # 只看最后一个位置的 logits
                logits = logits[:, -1, :] / temperature  # (B, V)
                
                # Top-k sampling (可选)
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Softmax + sample
                probs = F.softmax(logits, dim=-1)  # (B, V)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, idx_next], dim=1)  # (B, T+1)
        
        return input_ids


class TransformerBlock(nn.Module):
    """
    标准 Transformer Block: Self-Attention + FFN
    """
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.ln1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Feed-forward network
        self.ln2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
        
        Returns:
            x: (batch_size, seq_len, hidden_size)
        """
        # Self-attention with residual
        x = x + self.attn(self.ln1(x))
        
        # FFN with residual
        x = x + self.mlp(self.ln2(x))
        
        return x


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) multi-head self-attention
    """
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Q, K, V projections
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask (will be registered as buffer)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024),
            persistent=False
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
        
        Returns:
            x: (batch_size, seq_len, hidden_size)
        """
        B, T, H = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*H)
        q, k, v = qkv.split(self.hidden_size, dim=2)  # each (B, T, H)
        
        # Reshape to (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with causal mask
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        
        # Apply causal mask
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0,
            float('-inf')
        )
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, nh, T, T)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = attn_weights @ v  # (B, nh, T, hd)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, H)
        
        # Output projection
        output = self.out_proj(attn_output)
        output = self.resid_dropout(output)
        
        return output


def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试
    config = {
        'hidden_size': 512,
        'num_layers': 6,
        'num_heads': 8,
        'vocab_size': 200,
        'max_seq_len': 256,
        'dropout': 0.1
    }
    
    model = DecoderOnlyTransformer(config)
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 测试 forward
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, config['vocab_size'], (batch_size, seq_len))
    
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config['vocab_size'])
    
    # 测试生成
    prompt = torch.randint(0, config['vocab_size'], (1, 10))
    generated = model.generate(prompt, max_new_tokens=20)
    print(f"Generated shape: {generated.shape}")
    
    print("✅ 所有测试通过！")
