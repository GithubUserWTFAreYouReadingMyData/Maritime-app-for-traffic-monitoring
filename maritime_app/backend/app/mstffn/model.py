import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V):
        batch_size, seq_len, _ = Q.size()

        Q = self.w_q(Q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(K).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(V).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.w_o(attn_output)

        return output, attn_weights


class PositionalTimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.time_embedding = nn.Linear(3, d_model)

    def forward(self, x, time_features):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        time_emb = self.time_embedding(time_features).unsqueeze(1)
        x = x + time_emb
        return self.dropout(x)


class MSTFFN(nn.Module):
    def __init__(self, d_model=128, n_heads=8, n_layers=4, time_scales=None, dropout=0.1):
        super().__init__()
        if time_scales is None:
            time_scales = {"low": 24, "medium": 72, "high": 168}
        self.time_scales = time_scales
        self.d_model = d_model

        self.low_embedding = nn.Linear(time_scales["low"], d_model)
        self.medium_embedding = nn.Linear(time_scales["medium"], d_model)
        self.high_embedding = nn.Linear(time_scales["high"], d_model)

        self.pos_time_encoding = PositionalTimeEncoding(d_model, dropout)
        self.attention_layers = nn.ModuleList([MultiHeadAttention(d_model, n_heads, dropout) for _ in range(n_layers)])

        self.ff_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 4, d_model),
                )
                for _ in range(n_layers)
            ]
        )

        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.fusion_layer = nn.Sequential(nn.Linear(d_model * 3, d_model * 2), nn.ReLU(), nn.Dropout(dropout))

        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_model, 2)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, multiscale_data, time_features):
        low_emb = self.low_embedding(multiscale_data["low"]).unsqueeze(1)
        medium_emb = self.medium_embedding(multiscale_data["medium"]).unsqueeze(1)
        high_emb = self.high_embedding(multiscale_data["high"]).unsqueeze(1)

        low_emb = self.pos_time_encoding(low_emb, time_features)
        medium_emb = self.pos_time_encoding(medium_emb, time_features)
        high_emb = self.pos_time_encoding(high_emb, time_features)

        for i in range(len(self.attention_layers)):
            low_attn, _ = self.attention_layers[i](low_emb, low_emb, low_emb)
            low_emb = self.norm1[i](low_emb + self.dropout(low_attn))
            low_ff = self.ff_layers[i](low_emb)
            low_emb = self.norm2[i](low_emb + self.dropout(low_ff))

            medium_attn, _ = self.attention_layers[i](medium_emb, medium_emb, medium_emb)
            medium_emb = self.norm1[i](medium_emb + self.dropout(medium_attn))
            medium_ff = self.ff_layers[i](medium_emb)
            medium_emb = self.norm2[i](medium_emb + self.dropout(medium_ff))

            high_attn, _ = self.attention_layers[i](high_emb, high_emb, high_emb)
            high_emb = self.norm1[i](high_emb + self.dropout(high_attn))
            high_ff = self.ff_layers[i](high_emb)
            high_emb = self.norm2[i](high_emb + self.dropout(high_ff))

        low_pool = low_emb.mean(dim=1)
        medium_pool = medium_emb.mean(dim=1)
        high_pool = high_emb.mean(dim=1)

        concatenated = torch.cat([low_pool, medium_pool, high_pool], dim=1)
        fused = self.fusion_layer(concatenated)
        gaussian_params = self.prediction_head(fused)

        mu = gaussian_params[:, 0]
        sigma = torch.exp(gaussian_params[:, 1])

        return mu, sigma
