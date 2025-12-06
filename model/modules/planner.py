"""
Planner Module - High-level strategic reasoning

This module generates high-level plans for code changes, including:
- Intents (FIX, ADD, REFACTOR, etc.)
- Targets (functions, types, interfaces)
- Step-by-step reasoning

Based on HRM architecture: https://arxiv.org/abs/2506.21734
"""

import torch
import torch.nn as nn
from typing import Optional


class PlannerModule(nn.Module):
    """
    High-level planning module for hierarchical reasoning.

    Operates at the abstract level: understands what needs to be done
    but not the detailed implementation.

    Architecture: Standard transformer encoder
    Input: Problem description + Context
    Output: Plan embedding (intents, targets, steps)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.n_embd
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            config.n_positions,
            config.n_embd
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=config.n_embd * 4,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layer,
            norm=nn.LayerNorm(config.n_embd),
        )

        # Output head for plan token prediction
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Tie weights between embedding and lm_head (standard practice)
        self.lm_head.weight = self.token_embedding.weight

        # Layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights following GPT-2 conventions."""
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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass through planner module.

        Args:
            input_ids: [batch_size, seq_len] - Problem + context tokens
            attention_mask: [batch_size, seq_len] - Attention mask (1=attend, 0=ignore)
            position_ids: [batch_size, seq_len] - Position indices

        Returns:
            plan_logits: [batch_size, seq_len, vocab_size] - Plan token predictions
            plan_hidden: [batch_size, seq_len, n_embd] - Plan embeddings (for generator)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # Token + position embeddings
        token_embeds = self.token_embedding(input_ids)
        position_embeds = self.position_embedding(position_ids)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)

        # Convert attention mask to transformer format
        # PyTorch transformer expects: 0=attend, -inf=ignore (opposite of HF)
        if attention_mask is not None:
            # Convert from [batch, seq] to [batch, seq] with -inf for masked
            attention_mask = (1.0 - attention_mask) * -1e9

        # Transformer encoding
        hidden_states = self.transformer(
            hidden_states,
            mask=None,  # Causal mask not needed for encoder
            src_key_padding_mask=attention_mask,
        )

        # Layer norm
        hidden_states = self.ln_f(hidden_states)

        # Project to vocabulary for plan token prediction
        plan_logits = self.lm_head(hidden_states)

        return plan_logits, hidden_states

    def generate_plan(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
    ):
        """
        Generate plan tokens autoregressively.

        Args:
            input_ids: [batch_size, seq_len] - Problem + context
            max_new_tokens: Maximum plan tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            plan_tokens: [batch_size, seq_len + max_new_tokens] - Generated plan
        """
        self.eval()

        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = self.forward(input_ids)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)

            # Stop if we generated </PLAN> token (you'd check this properly)
            # For now, just continue

        return input_ids


if __name__ == "__main__":
    # Test the planner module
    from model.config import ModuleConfig

    config = ModuleConfig(
        vocab_size=50000,
        n_positions=512,
        n_embd=256,
        n_layer=3,
        n_head=8,
        dropout=0.1,
    )

    planner = PlannerModule(config)

    # Test input
    batch_size = 2
    seq_len = 128
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits, hidden = planner(input_ids)

    print(f"Planner Module Test:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Output hidden shape: {hidden.shape}")
    print(f"  Parameters: {sum(p.numel() for p in planner.parameters()) / 1e6:.2f}M")
