"""
Refinement Controller - Recursive refinement decision-making

This module decides whether to:
- CONTINUE (0): Syntax error, try again
- REFINE (1): Partial solution, improve the plan
- DONE (2): Solution is correct, stop

Based on TRM recursive reasoning: https://arxiv.org/abs/2510.04871
"""

import torch
import torch.nn as nn
from enum import IntEnum


class RefinementDecision(IntEnum):
    """Refinement decision types."""
    CONTINUE = 0  # Syntax error or invalid, regenerate with same plan
    REFINE = 1    # Partial solution, update plan and try again
    DONE = 2      # Solution is correct, stop iterating


class RefinementController(nn.Module):
    """
    Controls recursive refinement process.

    Takes plan embedding and validation feedback, decides next action.

    Small classifier: takes plan representation and outputs 3-way decision.
    """

    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Simple MLP classifier
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 3)  # 3-way classification

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        plan_hidden: torch.Tensor,
        validation_feedback: torch.Tensor = None,
    ):
        """
        Decide refinement action.

        Args:
            plan_hidden: [batch_size, plan_len, hidden_dim] - Plan embeddings
            validation_feedback: [batch_size, feedback_dim] - Optional validation signals

        Returns:
            decision_logits: [batch_size, 3] - Logits for CONTINUE/REFINE/DONE
        """
        # Pool plan hidden states (use last token or mean)
        # Here we use mean pooling
        plan_repr = plan_hidden.mean(dim=1)  # [batch_size, hidden_dim]

        # Optionally incorporate validation feedback
        if validation_feedback is not None:
            # Simple concatenation (could be more sophisticated)
            plan_repr = plan_repr + validation_feedback

        # MLP classification
        hidden = self.fc1(plan_repr)
        hidden = self.activation(hidden)
        hidden = self.dropout(hidden)
        decision_logits = self.fc2(hidden)

        return decision_logits

    def predict(self, plan_hidden: torch.Tensor, validation_feedback: torch.Tensor = None):
        """
        Get predicted decision (argmax).

        Returns:
            decision: [batch_size] - Predicted decision (0, 1, or 2)
        """
        logits = self.forward(plan_hidden, validation_feedback)
        decision = torch.argmax(logits, dim=-1)
        return decision

    def should_continue(self, plan_hidden: torch.Tensor, validation_feedback: torch.Tensor = None):
        """Check if we should continue refining (not DONE)."""
        decision = self.predict(plan_hidden, validation_feedback)
        return (decision != RefinementDecision.DONE).any()


class ValidationFeedbackEncoder(nn.Module):
    """
    Encodes validation feedback into embeddings.

    Validation signals:
    - Syntax OK/Error
    - Test Pass/Fail
    - Error messages (optional)
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Binary features: syntax_ok, tests_pass
        self.binary_proj = nn.Linear(2, hidden_dim // 2)

        # Optional: Error type embedding (if we categorize errors)
        self.error_embedding = nn.Embedding(10, hidden_dim // 2)  # 10 error types

        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        syntax_ok: torch.Tensor,
        tests_pass: torch.Tensor = None,
        error_type: torch.Tensor = None,
    ):
        """
        Encode validation feedback.

        Args:
            syntax_ok: [batch_size] - Boolean, is syntax valid?
            tests_pass: [batch_size] - Boolean, do tests pass?
            error_type: [batch_size] - Integer, error category (0-9)

        Returns:
            feedback_emb: [batch_size, hidden_dim] - Encoded feedback
        """
        batch_size = syntax_ok.size(0)

        # Binary features
        if tests_pass is None:
            tests_pass = torch.zeros_like(syntax_ok)

        binary_features = torch.stack([
            syntax_ok.float(),
            tests_pass.float()
        ], dim=1)  # [batch_size, 2]

        binary_emb = self.binary_proj(binary_features)  # [batch_size, hidden_dim//2]

        # Error type embedding
        if error_type is None:
            error_type = torch.zeros(batch_size, dtype=torch.long, device=syntax_ok.device)

        error_emb = self.error_embedding(error_type)  # [batch_size, hidden_dim//2]

        # Concatenate
        feedback_emb = torch.cat([binary_emb, error_emb], dim=1)  # [batch_size, hidden_dim]

        # Final projection
        feedback_emb = self.fc(feedback_emb)
        feedback_emb = self.activation(feedback_emb)

        return feedback_emb


if __name__ == "__main__":
    # Test refinement controller
    print("=== Refinement Controller Test ===")

    controller = RefinementController(hidden_dim=256)

    # Test input: plan hidden states
    batch_size = 4
    plan_len = 128
    hidden_dim = 256

    plan_hidden = torch.randn(batch_size, plan_len, hidden_dim)

    # Forward pass
    decision_logits = controller(plan_hidden)
    decisions = controller.predict(plan_hidden)

    print(f"  Plan hidden shape: {plan_hidden.shape}")
    print(f"  Decision logits shape: {decision_logits.shape}")
    print(f"  Predicted decisions: {decisions}")
    print(f"  Decision meanings:")
    print(f"    {RefinementDecision.CONTINUE}: CONTINUE (regenerate)")
    print(f"    {RefinementDecision.REFINE}: REFINE (update plan)")
    print(f"    {RefinementDecision.DONE}: DONE (solution correct)")
    print(f"  Parameters: {sum(p.numel() for p in controller.parameters()) / 1e3:.2f}K")

    print("\n=== Validation Feedback Encoder Test ===")

    feedback_encoder = ValidationFeedbackEncoder(hidden_dim=256)

    # Test validation feedback
    syntax_ok = torch.tensor([True, False, True, False])
    tests_pass = torch.tensor([True, False, False, True])
    error_type = torch.tensor([0, 2, 1, 3])  # Different error types

    feedback_emb = feedback_encoder(syntax_ok, tests_pass, error_type)

    print(f"  Syntax OK: {syntax_ok}")
    print(f"  Tests pass: {tests_pass}")
    print(f"  Error types: {error_type}")
    print(f"  Feedback embedding shape: {feedback_emb.shape}")

    # Test with feedback
    decision_with_feedback = controller(plan_hidden, feedback_emb)
    print(f"  Decision with feedback: {decision_with_feedback.argmax(dim=-1)}")

    print(f"\n  Total refinement params: {sum(p.numel() for p in controller.parameters()) / 1e3:.2f}K")
    print(f"  Total feedback encoder params: {sum(p.numel() for p in feedback_encoder.parameters()) / 1e3:.2f}K")
