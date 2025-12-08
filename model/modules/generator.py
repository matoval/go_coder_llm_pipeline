"""
Generator Module - Low-level code generation

This module generates actual Go code based on the high-level plan from the planner.
It uses cross-attention to the plan embeddings while generating code.

Key features:
- Causal (autoregressive) generation
- Cross-attention to planner output
- Generates detailed Go syntax

Based on HRM architecture: https://arxiv.org/abs/2506.21734
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (for autoregressive generation)."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Query, Key, Value projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # Causal mask (lower triangular)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.n_positions, config.n_positions))
            .view(1, 1, config.n_positions, config.n_positions)
        )

    def forward(self, x):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Reshape for multi-head attention
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Attention (Q * K^T / sqrt(d_k))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum of values
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class CrossAttention(nn.Module):
    """Multi-head cross-attention to planner embeddings."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Query from generator, Key/Value from planner
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, plan_hidden):
        """
        Cross-attend to plan embeddings.

        Args:
            x: [B, T_code, C] - Generator hidden states
            plan_hidden: [B, T_plan, C] - Planner embeddings

        Returns:
            y: [B, T_code, C] - Cross-attended output
        """
        B, T, C = x.size()
        _, T_plan, _ = plan_hidden.size()

        # Q from generator, K/V from planner
        q = self.q_proj(x)
        k = self.k_proj(plan_hidden)
        v = self.v_proj(plan_hidden)

        # Reshape for multi-head
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T_plan, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T_plan, self.n_head, C // self.n_head).transpose(1, 2)

        # Cross-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = torch.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Weighted sum
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """Feed-forward network."""

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class GeneratorBlock(nn.Module):
    """
    Transformer decoder block with:
    1. Causal self-attention (for autoregressive generation)
    2. Cross-attention to planner (key innovation!)
    3. Feed-forward network
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, plan_hidden):
        # Self-attention (causal)
        x = x + self.attn(self.ln_1(x))

        # Cross-attention to plan (KEY INNOVATION!)
        x = x + self.cross_attn(self.ln_2(x), plan_hidden)

        # Feed-forward
        x = x + self.mlp(self.ln_3(x))

        return x


class GeneratorModule(nn.Module):
    """
    Low-level code generation module.

    Generates actual Go code tokens autoregressively,
    conditioned on the high-level plan from PlannerModule.

    Architecture: Transformer decoder with cross-attention
    Input: Previous code tokens + Plan embeddings
    Output: Next code token predictions
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # Position embeddings
        self.position_embedding = nn.Embedding(config.n_positions, config.n_embd)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer decoder blocks (with cross-attention)
        self.blocks = nn.ModuleList([
            GeneratorBlock(config) for _ in range(config.n_layer)
        ])

        # Layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        plan_hidden: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through generator.

        Args:
            input_ids: [batch_size, seq_len] - Code tokens
            plan_hidden: [batch_size, plan_len, n_embd] - Plan embeddings from planner
            position_ids: [batch_size, seq_len] - Position indices

        Returns:
            logits: [batch_size, seq_len, vocab_size] - Next token predictions
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Position IDs
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # Transformer blocks (with cross-attention to plan)
        for block in self.blocks:
            hidden_states = block(hidden_states, plan_hidden)

        # Final layer norm
        hidden_states = self.ln_f(hidden_states)

        # Project to vocabulary
        logits = self.lm_head(hidden_states)

        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        plan_hidden: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
        eos_token_id: int = 2,
    ):
        """
        Generate code tokens autoregressively.

        Args:
            idx: [B, T] - Starting tokens
            plan_hidden: [B, plan_len, C] - Plan embeddings
            max_new_tokens: How many tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            eos_token_id: End-of-sequence token ID (default: 2)

        Returns:
            idx: [B, T + max_new_tokens] - Generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop to max context length
            idx_cond = idx if idx.size(1) <= self.config.n_positions else idx[:, -self.config.n_positions:]

            # Forward
            logits = self.forward(idx_cond, plan_hidden)

            # Get last token logits
            logits = logits[:, -1, :] / temperature

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat([idx, idx_next], dim=1)

            # Early stopping: check if we generated EOS token
            if (idx_next == eos_token_id).any():
                break

        return idx


if __name__ == "__main__":
    # Test the generator module
    from model.config import ModuleConfig

    config = ModuleConfig(
        vocab_size=50000,
        n_positions=1024,
        n_embd=256,
        n_layer=3,
        n_head=8,
        dropout=0.1,
    )

    generator = GeneratorModule(config)

    # Test inputs
    batch_size = 2
    code_len = 256
    plan_len = 128

    code_ids = torch.randint(0, config.vocab_size, (batch_size, code_len))
    plan_hidden = torch.randn(batch_size, plan_len, config.n_embd)

    # Forward pass
    logits = generator(code_ids, plan_hidden)

    print(f"Generator Module Test:")
    print(f"  Code input shape: {code_ids.shape}")
    print(f"  Plan hidden shape: {plan_hidden.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Parameters: {sum(p.numel() for p in generator.parameters()) / 1e6:.2f}M")

    # Test generation
    start_tokens = torch.randint(0, config.vocab_size, (1, 10))
    generated = generator.generate(start_tokens, plan_hidden[:1], max_new_tokens=20)
    print(f"  Generation test: {start_tokens.shape} -> {generated.shape}")
