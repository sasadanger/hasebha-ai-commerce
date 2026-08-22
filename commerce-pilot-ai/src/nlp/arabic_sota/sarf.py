"""Clean SARF (multi-view surface/stem/root, shared MARBERTv2, cross-attention, CNN+BiLSTM)
reproduction, matching the architecture verified directly from the audited SARF repo code
(see reports/generated/arabic_sota/sarf_protocol_lock.json for exact source-derived values).
This is an independent implementation, not a copy of the third-party notebook code -- built
using this project's own genuine paper-faithful/group-safe validation splits, per the audit
resolution's decision (SARF_SAFE_TO_BUILD_GROUP_SAFE_VERSION=YES via clean implementation).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class CrossAttention(nn.Module):
    """Surface as query, other view as key/value -- matches sarf_protocol_lock.json's
    cross_attention spec (paper Eq. 4-5)."""

    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(query, kv, kv)
        return out


class SARFModel(nn.Module):
    def __init__(self, checkpoint: str, num_classes: int = 3,
                 cnn_filters: int = 200, kernel_sizes=(3, 4, 5),
                 lstm_hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(checkpoint)  # SHARED across all 3 views
        hidden_size = self.encoder.config.hidden_size

        self.cross_attn_stem = CrossAttention(hidden_size)
        self.cross_attn_root = CrossAttention(hidden_size)

        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, cnn_filters, kernel_size=k, padding=k // 2) for k in kernel_sizes
        ])
        self.bilstm = nn.LSTM(hidden_size, lstm_hidden_dim, batch_first=True, bidirectional=True)

        fused_dim = cnn_filters * len(kernel_sizes) + lstm_hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(fused_dim, num_classes)

    def _encode(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def forward(self, surface, stem, root, labels=None):
        """surface/stem/root are dicts with 'input_ids'/'attention_mask' tensors."""
        H_surface = self._encode(**surface)
        H_stem = self._encode(**stem)
        H_root = self._encode(**root)

        H_stem_tilde = self.cross_attn_stem(H_surface, H_stem)
        H_root_tilde = self.cross_attn_root(H_surface, H_root)
        H_fused = (H_surface + H_stem_tilde + H_root_tilde) / 3.0  # paper Eq. 6, mean fusion

        # CNN branch operates on SURFACE encoding (verified from audited code, not H_fused)
        cnn_input = H_surface.transpose(1, 2)  # (B, hidden, T)
        cnn_feats = []
        for conv in self.convs:
            x = F.relu(conv(cnn_input))
            x = F.max_pool1d(x, kernel_size=x.size(2)).squeeze(2)
            cnn_feats.append(x)
        cnn_out = torch.cat(cnn_feats, dim=1)

        # BiLSTM branch operates on FUSED representation (verified from audited code)
        lstm_out, _ = self.bilstm(H_fused)
        lstm_out = lstm_out.mean(dim=1)

        fused = torch.cat([cnn_out, lstm_out], dim=1)
        logits = self.fc(self.dropout(fused))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)  # plain CE, matches audited code
        return {"loss": loss, "logits": logits}
