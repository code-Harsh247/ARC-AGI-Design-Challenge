"""
ARC-AGI Model Architecture: Iterative Refinement Transformer.

Components:
  - RoPE3D: 3-axis rotary positional encoding (row, col, pair_index)
  - ContextEncoder: bidirectional transformer over demo pairs + test input
  - IterativeDecoder: cross-attending decoder with shared weights, runs T times
  - ARCModel: full model wrapper
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import VOCAB_SIZE, PAD_TOKEN, NUM_COLORS


# ---------------------------------------------------------------------------
# 3D Rotary Positional Encoding
# ---------------------------------------------------------------------------
class RoPE3D(nn.Module):
    """
    3-axis Rotary Positional Encoding for (row, col, pair_index).

    Splits the model dimension into 3 equal chunks and applies standard 1D RoPE
    independently to each axis. This naturally encodes relative spatial distances
    along rows, columns, and across demo pairs.

    For dimensions not divisible by 3, the remainder goes to the row axis.
    """

    def __init__(self, d_model: int, max_positions: int = 64, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        # Split dims: row gets extra if not divisible by 3
        d_per_axis = d_model // 3
        self.d_row = d_model - 2 * d_per_axis
        self.d_col = d_per_axis
        self.d_pair = d_per_axis

        # Precompute inverse frequency bands for each axis
        self.register_buffer("inv_freq_row", self._make_inv_freq(self.d_row, base))
        self.register_buffer("inv_freq_col", self._make_inv_freq(self.d_col, base))
        self.register_buffer("inv_freq_pair", self._make_inv_freq(self.d_pair, base))

    @staticmethod
    def _make_inv_freq(dim: int, base: float) -> torch.Tensor:
        """Create inverse frequency bands for RoPE. dim must be even for pairing."""
        # Use dim // 2 pairs; if dim is odd, we'll handle the extra dim as unrotated
        half = dim // 2
        if half == 0:
            return torch.zeros(1)
        inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / dim))
        return inv_freq

    def _apply_rope_1d(self, x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor) -> torch.Tensor:
        """
        Apply 1D RoPE to a slice of the hidden dimension.

        Args:
            x: (..., D) tensor
            positions: (...,) integer positions
            inv_freq: (D//2,) inverse frequency bands
        Returns:
            (..., D) tensor with RoPE applied
        """
        D = x.shape[-1]
        half = D // 2
        if half == 0:
            return x

        # positions: (...) -> (..., 1) * inv_freq: (half,) -> (..., half)
        freqs = positions.unsqueeze(-1).float() * inv_freq  # (..., half)
        cos_f = freqs.cos()
        sin_f = freqs.sin()

        x1 = x[..., :half]
        x2 = x[..., half:2 * half]
        rotated = torch.cat([x1 * cos_f - x2 * sin_f, x2 * cos_f + x1 * sin_f], dim=-1)

        # If D is odd, append the last unrotated dimension
        if D > 2 * half:
            rotated = torch.cat([rotated, x[..., 2 * half:]], dim=-1)
        return rotated

    def forward(
        self,
        x: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply 3D RoPE to input tensor.

        Args:
            x: (B, L, D) hidden states
            rows: (B, L) row indices
            cols: (B, L) column indices
            pairs: (B, L) pair indices
        Returns:
            (B, L, D) with rotary encoding applied
        """
        d1 = self.d_row
        d2 = d1 + self.d_col

        x_row = self._apply_rope_1d(x[..., :d1], rows, self.inv_freq_row)
        x_col = self._apply_rope_1d(x[..., d1:d2], cols, self.inv_freq_col)
        x_pair = self._apply_rope_1d(x[..., d2:], pairs, self.inv_freq_pair)

        return torch.cat([x_row, x_col, x_pair], dim=-1)


def apply_rope_to_qk(
    rope: RoPE3D,
    q: torch.Tensor,
    k: torch.Tensor,
    q_rows: torch.Tensor,
    q_cols: torch.Tensor,
    q_pairs: torch.Tensor,
    k_rows: torch.Tensor,
    k_cols: torch.Tensor,
    k_pairs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE per-head to query and key tensors.

    Args:
        q, k: (B, n_heads, L, d_head)
        *_rows, *_cols, *_pairs: (B, L) position indices
    """
    B, H, L_q, D = q.shape
    _, _, L_k, _ = k.shape

    # Merge batch and head dims: (B*H, L, d_head)
    q_flat = q.reshape(B * H, L_q, D)
    k_flat = k.reshape(B * H, L_k, D)

    # Expand positions for all heads: (B, L) -> (B*H, L)
    q_rows_exp = q_rows.unsqueeze(1).expand(-1, H, -1).reshape(B * H, L_q)
    q_cols_exp = q_cols.unsqueeze(1).expand(-1, H, -1).reshape(B * H, L_q)
    q_pairs_exp = q_pairs.unsqueeze(1).expand(-1, H, -1).reshape(B * H, L_q)
    k_rows_exp = k_rows.unsqueeze(1).expand(-1, H, -1).reshape(B * H, L_k)
    k_cols_exp = k_cols.unsqueeze(1).expand(-1, H, -1).reshape(B * H, L_k)
    k_pairs_exp = k_pairs.unsqueeze(1).expand(-1, H, -1).reshape(B * H, L_k)

    q_rot = rope(q_flat, q_rows_exp, q_cols_exp, q_pairs_exp)
    k_rot = rope(k_flat, k_rows_exp, k_cols_exp, k_pairs_exp)

    q_out = q_rot.reshape(B, H, L_q, D)
    k_out = k_rot.reshape(B, H, L_k, D)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Transformer Building Blocks
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Multi-head attention with support for 3D RoPE and optional cross-attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: Optional[RoPE3D] = None,
        x_rows: Optional[torch.Tensor] = None,
        x_cols: Optional[torch.Tensor] = None,
        x_pairs: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
        kv_rows: Optional[torch.Tensor] = None,
        kv_cols: Optional[torch.Tensor] = None,
        kv_pairs: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L_q, D) query source
            kv: (B, L_k, D) key/value source (if None, self-attention)
            key_padding_mask: (B, L_k) True = valid, False = pad
            rope + position tensors for Q and K
        """
        B, L_q, _ = x.shape

        if kv is None:
            kv = x
            kv_rows = x_rows
            kv_cols = x_cols
            kv_pairs = x_pairs

        L_k = kv.shape[1]

        q = self.q_proj(x).view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(kv).view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(kv).view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE to Q and K
        if rope is not None and x_rows is not None:
            q, k = apply_rope_to_qk(
                rope, q, k,
                x_rows, x_cols, x_pairs,
                kv_rows, kv_cols, kv_pairs,
            )

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if key_padding_mask is not None:
            # key_padding_mask: (B, L_k), True = valid
            mask = ~key_padding_mask  # True = masked
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, L_q, d_head)
        out = out.transpose(1, 2).reshape(B, L_q, self.d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Standard FFN with SiLU (SwiGLU-style without the gate for simplicity)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: out = W2(SiLU(W_gate(x)) * W1(x))
        return self.dropout(self.w2(F.silu(self.w_gate(x)) * self.w1(x)))


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE3D,
        rows: torch.Tensor,
        cols: torch.Tensor,
        pairs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        h = self.norm1(x)
        x = x + self.attn(h, rope=rope, x_rows=rows, x_cols=cols, x_pairs=pairs, key_padding_mask=mask)
        # Pre-norm FFN
        h = self.norm2(x)
        x = x + self.ffn(h)
        return x


class DecoderBlock(nn.Module):
    """Pre-norm transformer decoder block with self-attention + cross-attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Self-attention
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # Cross-attention
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # FFN
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPE3D,
        x_rows: torch.Tensor,
        x_cols: torch.Tensor,
        x_pairs: torch.Tensor,
        encoder_out: torch.Tensor,
        enc_rows: torch.Tensor,
        enc_cols: torch.Tensor,
        enc_pairs: torch.Tensor,
        enc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention over decoder tokens
        h = self.norm1(x)
        x = x + self.self_attn(h, rope=rope, x_rows=x_rows, x_cols=x_cols, x_pairs=x_pairs)
        # Cross-attention to encoder output
        h = self.norm2(x)
        x = x + self.cross_attn(
            h, rope=rope,
            x_rows=x_rows, x_cols=x_cols, x_pairs=x_pairs,
            kv=encoder_out, kv_rows=enc_rows, kv_cols=enc_cols, kv_pairs=enc_pairs,
            key_padding_mask=enc_mask,
        )
        # FFN
        h = self.norm3(x)
        x = x + self.ffn(h)
        return x


# ---------------------------------------------------------------------------
# Context Encoder
# ---------------------------------------------------------------------------
class ContextEncoder(nn.Module):
    """
    Bidirectional transformer encoder over the full context sequence
    (all demo pairs + test input).

    Runs once per task, producing encoder output that the decoder cross-attends to.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN)
        self.rope = RoPE3D(d_model // n_heads)
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        rows: torch.Tensor,
        cols: torch.Tensor,
        pairs: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            tokens: (B, L) token IDs
            rows, cols, pairs: (B, L) position indices
            mask: (B, L) True = valid token
        Returns:
            (B, L, d_model) encoder hidden states
        """
        x = self.embedding(tokens)
        for layer in self.layers:
            x = layer(x, self.rope, rows, cols, pairs, mask)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Iterative Decoder
# ---------------------------------------------------------------------------
class IterativeDecoder(nn.Module):
    """
    Cross-attending decoder with shared weights, run T times for iterative refinement.

    At each step t:
      1. Takes the current output guess (embedded) + test input as decoder tokens
      2. Cross-attends to encoder output
      3. Produces per-cell logits over NUM_COLORS
      4. Updates the guess with argmax predictions for the next step
    """

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        # Embeddings for the decoder input (color guesses + test input colors)
        self.guess_embedding = nn.Embedding(NUM_COLORS, d_model)
        self.input_embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_TOKEN)
        # Learned type embeddings to distinguish test-input vs guess tokens
        self.type_embed = nn.Embedding(2, d_model)  # 0 = test input, 1 = guess
        self.rope = RoPE3D(d_model // n_heads)
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, NUM_COLORS, bias=False)

    def forward_one_step(
        self,
        guess_tokens: torch.Tensor,
        guess_rows: torch.Tensor,
        guess_cols: torch.Tensor,
        test_input_tokens: torch.Tensor,
        test_input_rows: torch.Tensor,
        test_input_cols: torch.Tensor,
        test_input_mask: torch.Tensor,
        encoder_out: torch.Tensor,
        enc_rows: torch.Tensor,
        enc_cols: torch.Tensor,
        enc_pairs: torch.Tensor,
        enc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Single decoder forward pass.

        Args:
            guess_tokens: (B, H*W) current color guesses for output cells
            guess_rows/cols: (B, H*W) spatial positions
            test_input_tokens: (B, L_in) test input cell values
            test_input_rows/cols: (B, L_in) spatial positions
            test_input_mask: (B, L_in) True = valid
            encoder_out: (B, L_ctx, D) from ContextEncoder
            enc_rows/cols/pairs: (B, L_ctx) positions
            enc_mask: (B, L_ctx) True = valid

        Returns:
            logits: (B, H*W, NUM_COLORS) per-cell color logits
        """
        B = guess_tokens.shape[0]
        L_guess = guess_tokens.shape[1]
        L_in = test_input_tokens.shape[1]

        # Embed and concatenate test input + guess
        inp_emb = self.input_embedding(test_input_tokens) + self.type_embed(
            torch.zeros(B, L_in, dtype=torch.long, device=guess_tokens.device)
        )
        guess_emb = self.guess_embedding(guess_tokens) + self.type_embed(
            torch.ones(B, L_guess, dtype=torch.long, device=guess_tokens.device)
        )

        # Concatenate: [test_input | guess]
        x = torch.cat([inp_emb, guess_emb], dim=1)  # (B, L_in + L_guess, D)

        # Pair index: use a high value to distinguish from demo pairs
        pair_idx_val = 5  # distinct from demo pair indices 0-4
        dec_rows = torch.cat([test_input_rows, guess_rows], dim=1)
        dec_cols = torch.cat([test_input_cols, guess_cols], dim=1)
        dec_pairs = torch.full((B, L_in + L_guess), pair_idx_val,
                               dtype=torch.long, device=guess_tokens.device)

        # Run decoder layers
        for layer in self.layers:
            x = layer(
                x, self.rope, dec_rows, dec_cols, dec_pairs,
                encoder_out, enc_rows, enc_cols, enc_pairs, enc_mask,
            )

        x = self.final_norm(x)

        # Extract only the guess positions (last L_guess tokens) and project to logits
        guess_hidden = x[:, L_in:, :]  # (B, L_guess, D)
        logits = self.output_head(guess_hidden)  # (B, L_guess, NUM_COLORS)
        return logits


# ---------------------------------------------------------------------------
# Per-Task Embeddings
# ---------------------------------------------------------------------------
class PerTaskEmbedding(nn.Module):
    """
    Learnable per-task vectors added to encoder input during training.

    At eval, either zero-initialize or retrieve nearest training task's embedding.
    """

    def __init__(self, n_tasks: int, d_model: int):
        super().__init__()
        self.embeddings = nn.Embedding(n_tasks, d_model)
        nn.init.normal_(self.embeddings.weight, std=0.02)

    def forward(self, task_indices: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Args:
            task_indices: (B,) task ID within training set
            seq_len: L — length of the context sequence
        Returns:
            (B, L, d_model) additive embedding
        """
        emb = self.embeddings(task_indices)  # (B, d_model)
        return emb.unsqueeze(1).expand(-1, seq_len, -1)


# ---------------------------------------------------------------------------
# Full ARC Model
# ---------------------------------------------------------------------------
class ARCModel(nn.Module):
    """
    Iterative Refinement Transformer for ARC-AGI.

    Architecture:
      1. ContextEncoder processes demo pairs + test input (runs once)
      2. IterativeDecoder refines output guess over T steps (shared weights)
      3. Deep supervision: return logits at every step

    Parameter budget (~28.5M for default config):
      - ContextEncoder (8L, d=384): ~14.2M
      - IterativeDecoder (6L, d=384): ~14.2M
      - PerTaskEmbedding (400, d=384): ~154K
      - Misc embeddings: ~10K
    """

    def __init__(
        self,
        d_model: int = 384,
        enc_layers: int = 8,
        dec_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1536,
        n_train_tasks: int = 400,
        dropout: float = 0.1,
        refine_steps: int = 5,
    ):
        super().__init__()
        self.d_model = d_model
        self.refine_steps = refine_steps

        self.encoder = ContextEncoder(d_model, n_heads, enc_layers, d_ff, dropout)
        self.decoder = IterativeDecoder(d_model, n_heads, dec_layers, d_ff, dropout)
        self.task_embed = PerTaskEmbedding(n_train_tasks, d_model)

        # Initialize weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_rows: torch.Tensor,
        context_cols: torch.Tensor,
        context_pairs: torch.Tensor,
        context_mask: torch.Tensor,
        target_rows: torch.Tensor,
        target_cols: torch.Tensor,
        target_mask: torch.Tensor,
        test_input_tokens: torch.Tensor,
        test_input_rows: torch.Tensor,
        test_input_cols: torch.Tensor,
        test_input_mask: torch.Tensor,
        task_idx: Optional[torch.Tensor] = None,
        T: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Full forward pass with iterative refinement.

        Returns:
            List of T tensors, each (B, L_out, NUM_COLORS) logits
        """
        if T is None:
            T = self.refine_steps

        B = context_tokens.shape[0]
        L_out = target_rows.shape[1]

        # --- Encode context (once) ---
        enc_input = self.encoder.embedding(context_tokens)

        # Add per-task embedding if task_idx provided (training mode)
        if task_idx is not None:
            enc_input = enc_input + self.task_embed(task_idx, enc_input.shape[1])

        # Run encoder layers on pre-embedded input
        x = enc_input
        for layer in self.encoder.layers:
            x = layer(x, self.encoder.rope, context_rows, context_cols, context_pairs, context_mask)
        encoder_out = self.encoder.final_norm(x)

        # --- Iterative decoder ---
        # Initial guess: all zeros (black)
        guess = torch.zeros(B, L_out, dtype=torch.long, device=context_tokens.device)

        all_logits = []
        for t in range(T):
            logits = self.decoder.forward_one_step(
                guess_tokens=guess,
                guess_rows=target_rows,
                guess_cols=target_cols,
                test_input_tokens=test_input_tokens,
                test_input_rows=test_input_rows,
                test_input_cols=test_input_cols,
                test_input_mask=test_input_mask,
                encoder_out=encoder_out,
                enc_rows=context_rows,
                enc_cols=context_cols,
                enc_pairs=context_pairs,
                enc_mask=context_mask,
            )
            all_logits.append(logits)
            # Update guess for next step (stop gradient — no backprop through argmax)
            guess = logits.argmax(dim=-1).detach()

        return all_logits

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict:
        """Detailed parameter count by component."""
        breakdown = {}
        for name, module in self.named_children():
            count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            breakdown[name] = count
        breakdown["total"] = self.count_parameters()
        return breakdown


# ---------------------------------------------------------------------------
# Debug / Small Model Factory
# ---------------------------------------------------------------------------
def make_debug_model(n_train_tasks: int = 400) -> ARCModel:
    """~5M param model for quick pipeline validation."""
    return ARCModel(
        d_model=128,
        enc_layers=4,
        dec_layers=3,
        n_heads=4,
        d_ff=512,
        n_train_tasks=n_train_tasks,
        dropout=0.1,
        refine_steps=3,
    )


def make_full_model(n_train_tasks: int = 400) -> ARCModel:
    """~28M param model — the real deal."""
    return ARCModel(
        d_model=384,
        enc_layers=8,
        dec_layers=6,
        n_heads=8,
        d_ff=1536,
        n_train_tasks=n_train_tasks,
        dropout=0.1,
        refine_steps=5,
    )
