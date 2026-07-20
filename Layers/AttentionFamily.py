import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from Layers.Masking import TriangleMask


def get_weight(last_dim, d_model, init_model="kaiming", requires_grad=True):
    weight_ma = torch.empty(last_dim, d_model)
    if init_model == "kaiming":
        nn.init.kaiming_normal_(weight_ma)
    elif init_model == "xavier":
        nn.init.xavier_normal_(weight_ma)
    else:
        nn.init.kaiming_normal_(weight_ma)
    weight_ma.requires_grad = requires_grad
    return weight_ma


# basic Attention class
class Attention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attn_dropout=0.1, output_attention=False):
        super(Attention, self).__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.attn_dropout = attn_dropout
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value):
        b, l, h, e = query.size()
        _, s, _, d = value.size()
        scale = self.scale if self.scale is not None else 1 / sqrt(e)

        #  De-stationary attention
        attn_score = torch.einsum("blhe, bshe->bhls", query, key) * scale

        if self.mask_flag:
            attn_mask = TriangleMask(shape=(b, l), device=query.device).mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_score = attn_score.masked_fill(attn_mask.mask, -np.inf)

        attn_weight = self.dropout(torch.softmax(attn_score, dim=-1))
        output = torch.einsum("bhls, bshd->blhd", attn_weight, value)

        if self.output_attention:
            return output, attn_weight
        else:
            return output, None


# DSAttention, usually for time series
class DSAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, tau=1., delta=0., attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.mask_flag = mask_flag
        self.scale = scale
        self.tau = tau
        self.delta = delta
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention

    def forward(self, query, key, value):
        b, l, h, e = query.size()
        _, s, _, d = value.size()
        scale = self.scale if self.scale is not None else 1 / sqrt(e)

        #  De-stationary attention
        attn_score = torch.einsum("blhe, bshe->bhls", query, key) * scale * self.tau + self.delta

        if self.mask_flag:
            attn_mask = TriangleMask(shape=(b, l), device=query.device).mask
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            attn_score = attn_score.masked_fill(attn_mask.mask, -np.inf)

        attn_weight = self.dropout(torch.softmax(attn_score, dim=-1))
        output = torch.einsum("bhls, bshd->blhd", attn_weight, value)

        if self.output_attention:
            return output, attn_weight
        else:
            return output, None


class CrossAttentionLayer(nn.Module):
    def __init__(self, n_head=1, scale=None, output_attention=False, d_model=1, init_model="kaiming"):
        super(CrossAttentionLayer, self).__init__()
        self.scale = scale
        self.d_model = d_model
        self.output_attention = output_attention
        self.d_head = d_model // n_head
        self.output_projection = nn.Linear(n_head*self.d_head, d_model)

        self.W_q = nn.Parameter(get_weight(d_model, self.d_head*n_head, init_model))
        self.W_k = nn.Parameter(get_weight(d_model, self.d_head*n_head, init_model))
        self.W_v = nn.Parameter(get_weight(d_model, self.d_head*n_head, init_model))

    def forward(self, query, key, value):
        """
            Cross Attention should satisfy key == value, which means
            query: [bs, n_vars, n_patches, d_model]
            key: [bs, key_bars, n_patches, d_model]
            value: [bs, key_vars, n_patches, d_model]
        """
        self.W_q = self.W_q.to(query.device)
        self.W_k = self.W_k.to(query.device)
        self.W_v = self.W_v.to(query.device)

        Q = query @ self.W_q
        K = key @ self.W_k
        V = value @ self.W_v

        bs, n, npa, d = Q.size()
        _, kn, _, _ = K.size()

        scale = 1 / sqrt(self.d_head) if self.scale is None else self.scale

        if self.d_head == self.d_model:
            # b: batch_size, q: q_vars, n: n_patches, d: d_model, k: k_vars, h: heads
            attn_score = torch.einsum("bqnd, bknd->bnqk", Q, K) * scale
            attn_weight = torch.softmax(attn_score, dim=-1)
            output = torch.einsum("bnqk, bknd->bqnd", attn_weight, V)

        else:
            Q = Q.reshape(bs, n, npa, -1, self.d_head)
            K = K.reshape(bs, kn, npa, -1, self.d_head)
            V = V.reshape(bs, kn, npa, -1, self.d_head)
            attn_score = torch.einsum("bqnhd, bknhd->bnhqk", Q, K)
            attn_weight = torch.softmax(attn_score, dim=-1)
            output = torch.einsum("bnhqk, bknhd->bqnhd", attn_weight, V)

            output = output.reshape(bs, n, npa, d)

        output = self.output_projection(output)
        if self.output_attention:
            return output, attn_weight
        else:
            return output, None


class SelfAttentionLayer(nn.Module):
    def __init__(self, n_head, attn_type, mask_flag, scale, tau, delta, attention_dropout=0.1, output_attention=False,
                 last_dim=1, d_model=1, init_model="kaiming"):
        super(SelfAttentionLayer, self).__init__()
        if attn_type == "DS":
            self.attention = DSAttention(mask_flag, scale, tau, delta, attention_dropout, output_attention)
        else:
            self.attention = Attention(mask_flag, scale, attention_dropout, output_attention)

        d_key = d_value = d_model // n_head

        self.W_q = nn.Parameter(get_weight(last_dim, d_key * n_head, init_model))
        self.W_k = nn.Parameter(get_weight(last_dim, d_key * n_head, init_model))
        self.W_v = nn.Parameter(get_weight(last_dim, d_value * n_head, init_model))
        self.n_head = n_head

        self.output_projection = nn.Linear(d_value * n_head, d_model)

    def forward(self, query):
        origin_dim = query.shape
        if len(origin_dim) == 4:
            query = query.reshape(origin_dim[0]*origin_dim[1], origin_dim[2], -1)
            key = query.reshape(origin_dim[0]*origin_dim[1], origin_dim[2], -1)
            value = query.reshape(origin_dim[0]*origin_dim[1], origin_dim[2], -1)
        else:
            key = value = query

        Q = query @ self.W_q
        K = key @ self.W_k
        V = value @ self.W_v

        b, n, d = Q.size()
        bv, nv, _ = V.size()
        Q = Q.view(b, n, self.n_head, -1)
        K = K.view(b, n, self.n_head, -1)
        V = V.view(bv, nv, self.n_head, -1)

        output, attn_weight = self.attention(Q, K, V)

        output = output.reshape(b, n, -1)
        output = self.output_projection(output)  # [bs*n_vars, n_patches, d_model]

        if len(origin_dim) == 4:
            output = output.reshape(origin_dim[0], origin_dim[1], origin_dim[2], -1)

        return output, attn_weight


if __name__ == "__main__":
    ca = CrossAttentionLayer(n_head=4, scale=None, output_attention=False, d_model=512, init_model="kaiming")
    q = torch.rand((64, 3, 27, 512))
    k = torch.rand((64, 12, 27, 512))
    v = k
    output, _ = ca(q, k, v)
    print(output.shape)
