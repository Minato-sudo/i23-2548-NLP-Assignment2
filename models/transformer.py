import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention with optional padding mask."""
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, q, k, v, mask=None):
        # q, k: (B, H, L, d_k), v: (B, H, L, d_v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        # Replace NaN (from all-inf rows) with 0
        attn = torch.nan_to_num(attn, nan=0.0)
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with h independent head projections."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        # Separate projection matrices per head (stored as ModuleLists)
        self.w_qs = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_heads)])
        self.w_ks = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_heads)])
        self.w_vs = nn.ModuleList([nn.Linear(d_model, self.d_k, bias=False) for _ in range(n_heads)])

        self.attention = ScaledDotProductAttention(self.d_k)
        self.out_proj = nn.Linear(d_model, d_model)  # shared output projection

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # Compute per-head attention
        head_outputs = []
        head_attns = []
        for h in range(self.n_heads):
            q = self.w_qs[h](x)  # (B, L, d_k)
            k = self.w_ks[h](x)
            v = self.w_vs[h](x)

            q = q.unsqueeze(1)  # (B, 1, L, d_k) for attention module
            k = k.unsqueeze(1)
            v = v.unsqueeze(1)

            head_out, attn = self.attention(q, k, v, mask)
            head_outputs.append(head_out.squeeze(1))  # (B, L, d_k)
            head_attns.append(attn.squeeze(1))         # (B, L, L)

        # Concatenate heads -> (B, L, d_model)
        concatenated = torch.cat(head_outputs, dim=-1)
        output = self.out_proj(concatenated)

        # Stack attention weights: (B, H, L, L)
        all_attn = torch.stack(head_attns, dim=1)
        return output, all_attn


class PositionWiseFeedForward(nn.Module):
    """Two-layer FFN: d_model -> d_ff -> d_model with ReLU."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding stored as a fixed (non-learned) buffer."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoderBlock(nn.Module):
    """Pre-Layer Norm encoder block: x <- x + Dropout(MultiHead(LN(x))), etc."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN self-attention
        x2, attn = self.attn(self.norm1(x), mask)
        x = x + self.dropout1(x2)
        # Pre-LN FFN
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x, attn


class TransformerClassifier(nn.Module):
    """Transformer encoder for topic classification with [CLS] token."""
    def __init__(self, vocab_size, d_model=128, n_heads=4, d_ff=512,
                 n_layers=4, n_classes=5, max_len=257, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Learned [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 4 stacked encoder blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # MLP classification head: d_model -> 64 -> n_classes
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Token embeddings
        x = self.embedding(x)  # (B, L, d_model)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        x = torch.cat([cls_tokens, x], dim=1)                   # (B, L+1, d_model)

        # Build attention mask for (B, 1, 1, L+1) shape
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)  # (B, L+1)
            attn_mask = full_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L+1)
        else:
            attn_mask = None

        x = self.pos_encoding(x)
        x = self.dropout(x)

        attentions = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask)
            attentions.append(attn)

        x = self.norm(x)
        cls_output = x[:, 0]           # (B, d_model)
        logits = self.classifier(cls_output)
        return logits, attentions
