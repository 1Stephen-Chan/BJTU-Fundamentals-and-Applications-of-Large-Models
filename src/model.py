import torch
import torch.nn as nn
import math


# ======================
# 1. Scaled Dot-Product Attention
# ======================
class ScaledDotProductAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out, attn


# ======================
# 2. Multi-Head Attention
# ======================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=128, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, Q, K, V, mask=None):
        B = Q.size(0)
        Q = self.W_Q(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        out, attn = self.attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        out = self.fc_out(out)
        return out


# ======================
# 3. Feed Forward
# ======================
class FeedForward(nn.Module):
    def __init__(self, d_model=128, d_ff=512, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


# ======================
# 4. Positional Encoding
# ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ======================
# 5. Encoder Layer
# ======================
class EncoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1,
                 use_residual=True, use_layernorm=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

    def forward(self, x, mask=None):
        attn_out = self.self_attn(x, x, x, mask)
        out1 = self.dropout(attn_out)
        if self.use_residual:
            out1 = x + out1
        if self.use_layernorm:
            out1 = self.norm1(out1)

        ffn_out = self.ffn(out1)
        out2 = self.dropout(ffn_out)
        if self.use_residual:
            out2 = out1 + out2
        if self.use_layernorm:
            out2 = self.norm2(out2)

        return out2


# ======================
# 6. Decoder Layer
# ======================
class DecoderLayer(nn.Module):
    def __init__(self, d_model=128, n_heads=4, d_ff=512, dropout=0.1,
                 use_residual=True, use_layernorm=True):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

    def forward(self, x, enc_out, tgt_mask=None, memory_mask=None):
        attn1 = self.self_attn(x, x, x, tgt_mask)
        out1 = self.dropout(attn1)
        if self.use_residual:
            out1 = x + out1
        if self.use_layernorm:
            out1 = self.norm1(out1)

        attn2 = self.cross_attn(out1, enc_out, enc_out, memory_mask)
        out2 = self.dropout(attn2)
        if self.use_residual:
            out2 = out1 + out2
        if self.use_layernorm:
            out2 = self.norm2(out2)

        ffn_out = self.ffn(out2)
        out3 = self.dropout(ffn_out)
        if self.use_residual:
            out3 = out2 + out3
        if self.use_layernorm:
            out3 = self.norm3(out3)

        return out3


# ======================
# 7. Transformer
# ======================
class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=128, n_heads=4, d_ff=512,
                 num_layers=2, dropout=0.1,
                 use_positional_encoding=True,
                 use_residual=True,
                 use_layernorm=True):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.use_positional_encoding = use_positional_encoding

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout,
                         use_residual=use_residual,
                         use_layernorm=use_layernorm)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout,
                         use_residual=use_residual,
                         use_layernorm=use_layernorm)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc = self.src_embed(src)
        if self.use_positional_encoding:
            enc = self.pos_enc(enc)
        for layer in self.encoder_layers:
            enc = layer(enc, src_mask)

        dec = self.tgt_embed(tgt)
        if self.use_positional_encoding:
            dec = self.pos_enc(dec)
        for layer in self.decoder_layers:
            dec = layer(dec, enc, tgt_mask)

        out = self.fc_out(dec)
        return out


# ======================
# 8. Mask 生成函数
# ======================
def generate_square_subsequent_mask(size):
    mask = torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0)
    return mask
