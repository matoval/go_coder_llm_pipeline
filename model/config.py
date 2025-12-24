"""
Hierarchical Recursive Go Coder (HRGC) Model Configuration

This module defines the configuration for the HRM-based architecture,
which consists of two modules:
1. Planning Module (high-level reasoning)
2. Generation Module (low-level implementation)

Based on research from:
- HRM: https://arxiv.org/abs/2506.21734
- TRM: https://arxiv.org/abs/2510.04871
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ModuleConfig:
    """Configuration for a single transformer module (planner or generator)."""

    vocab_size: int = 50000
    n_positions: int = 1024
    n_embd: int = 256
    n_layer: int = 3
    n_head: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    layer_norm_epsilon: float = 1e-5

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"


@dataclass
class HRMConfig:
    """
    Hierarchical Recursive Model Configuration.

    This configuration defines a two-module architecture with:
    - Planning Module: High-level strategic reasoning (abstract, slow)
    - Generation Module: Low-level implementation (detailed, fast)
    - Refinement Controller: Decides when to continue/refine/stop

    Total parameters: ~20-30M (6x smaller than GPT-2 125M)
    """

    # Shared configuration
    vocab_size: int = 50000
    bos_token_id: int = 1  # <s>
    eos_token_id: int = 2  # </s>
    pad_token_id: int = 3  # <pad> - FIXED: was 0, actual tokenizer uses 3

    # Planning Module Configuration (High-level reasoning)
    planner_config: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(
            vocab_size=50000,
            n_positions=512,  # Plan length (shorter than code)
            n_embd=256,  # Compact embedding
            n_layer=3,  # Lightweight
            n_head=8,
            dropout=0.1,
        )
    )

    # Generation Module Configuration (Low-level implementation)
    generator_config: ModuleConfig = field(
        default_factory=lambda: ModuleConfig(
            vocab_size=50000,
            n_positions=1024,  # Code length
            n_embd=256,  # Compact embedding
            n_layer=3,  # Lightweight
            n_head=8,
            dropout=0.1,
        )
    )

    # Cross-attention configuration
    cross_attention_dim: int = 256
    cross_attention_heads: int = 8

    # Recursive refinement configuration
    max_refinement_iterations: int = 5
    refinement_threshold: float = 0.9
    enable_validation: bool = True

    # Training configuration
    training_config: Dict[str, Any] = field(
        default_factory=lambda: {
            # Learning rates (different for each module)
            "learning_rate": 3e-4,
            "planner_lr_multiplier": 1.2,  # Planner trains slightly faster
            "generator_lr_multiplier": 0.8,  # Generator more conservative
            "refinement_lr": 3e-4,
            # Batch configuration
            "batch_size": 8,  # 4x larger than GPT-2 (smaller model)
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 32,
            # Training steps
            "max_steps": 125000,  # ~4B tokens (vs 10B for GPT-2)
            "warmup_steps": 1000,
            "weight_decay": 0.1,
            # Optimization
            "fp16": True,
            "gradient_checkpointing": False,  # Optional, low VRAM use
            # Saving & logging
            "save_steps": 1000,
            "eval_steps": 500,
            "logging_steps": 50,
            # HRM-specific
            "refinement_warmup": 5000,  # Start refinement training later
            "validation_frequency": 100,  # Run syntax checks every N steps
            # Loss weights
            "plan_loss_weight": 0.4,
            "code_loss_weight": 0.4,
            "refinement_loss_weight": 0.2,
        }
    )

    # Special tokens for hierarchical reasoning
    # FIXED: These now match the actual tokenizer token IDs from go_coder_llm.model
    special_tokens: Dict[str, int] = field(
        default_factory=lambda: {
            # Base tokens (MUST match token IDs above)
            "<unk>": 0,   # Unknown token
            "<s>": 1,     # Matches bos_token_id (BEGIN-OF-SEQUENCE)
            "</s>": 2,    # Matches eos_token_id (END-OF-SEQUENCE)
            "<pad>": 3,   # Matches pad_token_id
            # Structural tokens
            "<REPO>": 4,
            "<PR_TITLE>": 5,
            "<PR_BODY>": 6,
            "<FILE>": 7,
            "<BEFORE>": 8,
            "<AFTER>": 9,
            # Code and Planning tokens (CRITICAL for training)
            "<CODE>": 11,
            "<PLAN>": 13,
            "</PLAN>": 14,
            "<STEP>": 15,
            # Intent tokens
            "<INTENT:FIX>": 16,
            "<INTENT:ADD>": 17,
            "<INTENT:REFACTOR>": 18,
            "<INTENT:OPTIMIZE>": 19,
            "<INTENT:UPDATE>": 20,
            "<INTENT:REMOVE>": 21,
            # Target tokens
            "<TARGET:func>": 22,
            "<TARGET:type>": 23,
            "<TARGET:interface>": 24,
            "<TARGET:method>": 25,
            "<TARGET:var>": 26,
            # Validation tokens
            "<VALIDATE>": 27,
            "<SYNTAX_OK>": 28,
            "<TEST_PASS>": 30,
            # Context tokens
            "<CONTEXT>": 33,
            "<PROBLEM>": 34,
        }
    )

    def get_planner_params(self) -> int:
        """Calculate approximate parameter count for planner module."""
        cfg = self.planner_config
        # Embedding: vocab_size * n_embd
        embedding = cfg.vocab_size * cfg.n_embd
        # Position embedding: n_positions * n_embd
        pos_emb = cfg.n_positions * cfg.n_embd
        # Transformer layers: 12 * n_embd^2 * n_layer (approx)
        transformer = 12 * (cfg.n_embd**2) * cfg.n_layer
        # LM head: n_embd * vocab_size
        lm_head = cfg.n_embd * cfg.vocab_size
        return embedding + pos_emb + transformer + lm_head

    def get_generator_params(self) -> int:
        """Calculate approximate parameter count for generator module."""
        cfg = self.generator_config
        # Same calculation as planner
        embedding = cfg.vocab_size * cfg.n_embd
        pos_emb = cfg.n_positions * cfg.n_embd
        transformer = 12 * (cfg.n_embd**2) * cfg.n_layer
        # Add cross-attention parameters
        cross_attn = 4 * (cfg.n_embd**2) * cfg.n_layer
        lm_head = cfg.n_embd * cfg.vocab_size
        return embedding + pos_emb + transformer + cross_attn + lm_head

    def get_total_params(self) -> int:
        """Calculate total parameter count for HRM model."""
        planner = self.get_planner_params()
        generator = self.get_generator_params()
        # Refinement controller: small classifier
        refinement = 256 * 3  # Linear layer for 3-way classification
        return planner + generator + refinement

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model_type": "hrm",
            "vocab_size": self.vocab_size,
            "planner_config": {
                "n_positions": self.planner_config.n_positions,
                "n_embd": self.planner_config.n_embd,
                "n_layer": self.planner_config.n_layer,
                "n_head": self.planner_config.n_head,
                "dropout": self.planner_config.dropout,
            },
            "generator_config": {
                "n_positions": self.generator_config.n_positions,
                "n_embd": self.generator_config.n_embd,
                "n_layer": self.generator_config.n_layer,
                "n_head": self.generator_config.n_head,
                "dropout": self.generator_config.dropout,
            },
            "cross_attention_dim": self.cross_attention_dim,
            "max_refinement_iterations": self.max_refinement_iterations,
            "training_config": self.training_config,
            "total_params": self.get_total_params(),
        }

    @classmethod
    def from_pretrained(cls, path: str) -> "HRMConfig":
        """Load config from saved checkpoint."""
        import json

        with open(path, "r") as f:
            config_dict = json.load(f)

        # Reconstruct config
        config = cls()
        # Update fields from loaded config
        # (simplified - would need full deserialization logic)
        return config

    def __repr__(self) -> str:
        planner_params = self.get_planner_params() / 1e6
        generator_params = self.get_generator_params() / 1e6
        total_params = self.get_total_params() / 1e6

        return f"""HRMConfig(
  Planner:    {planner_params:.1f}M params ({self.planner_config.n_layer} layers, {self.planner_config.n_embd} dim)
  Generator:  {generator_params:.1f}M params ({self.generator_config.n_layer} layers, {self.generator_config.n_embd} dim)
  Total:      {total_params:.1f}M params
  Max Iters:  {self.max_refinement_iterations}
  Vocab Size: {self.vocab_size}
)"""


# Default configurations

def get_tiny_config() -> HRMConfig:
    """Tiny HRM model for testing (7-10M params)."""
    config = HRMConfig()
    config.planner_config = ModuleConfig(
        vocab_size=50000,
        n_positions=256,
        n_embd=128,
        n_layer=2,
        n_head=4,
    )
    config.generator_config = ModuleConfig(
        vocab_size=50000,
        n_positions=512,
        n_embd=128,
        n_layer=2,
        n_head=4,
    )
    # Match cross_attention_dim to actual embedding size
    config.cross_attention_dim = 128
    config.cross_attention_heads = 4
    return config


def get_small_config() -> HRMConfig:
    """Small HRM model (20-30M params) - DEFAULT."""
    return HRMConfig()  # Uses default values


def get_medium_config() -> HRMConfig:
    """Medium HRM model (60-80M params)."""
    config = HRMConfig()
    config.planner_config = ModuleConfig(
        vocab_size=50000,
        n_positions=512,
        n_embd=384,
        n_layer=5,
        n_head=12,
    )
    config.generator_config = ModuleConfig(
        vocab_size=50000,
        n_positions=1024,
        n_embd=384,
        n_layer=5,
        n_head=12,
    )
    return config


if __name__ == "__main__":
    # Test configurations
    print("=== Tiny Config ===")
    tiny = get_tiny_config()
    print(tiny)

    print("\n=== Small Config (DEFAULT) ===")
    small = get_small_config()
    print(small)

    print("\n=== Medium Config ===")
    medium = get_medium_config()
    print(medium)

    print("\n=== Config Dictionary ===")
    import json

    print(json.dumps(small.to_dict(), indent=2))
