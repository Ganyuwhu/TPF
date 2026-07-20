import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

from Utils.Dir import *
from Utils.Tools import Transpose
from Layers.Embed import PositionalEmbedding, PatchEmbedding
from Layers.AttentionFamily import SelfAttentionLayer, CrossAttentionLayer


class PollutantEncoding:
    """
        Sequence encoding based on the pollutant list
    """
    def __init__(self, pollutants=("NO2", "PM25", "O3")):
        self.encoding = {}
        dim = len(pollutants)
        one_hot = np.eye(dim)
        for i, pollutant in enumerate(pollutants):
            self.encoding[pollutant] = one_hot[i]

    def __call__(self):
        return self.encoding


class Preprocessing(nn.Module):
    """
        Preprocessing is used for two task:
        1. Turn static factors to time series
        2. Patch all time series
    """
    def __init__(self, seq_len, d_model, patch_len, stride, padding=0, dropout=0.1, dims=(1, 2)):
        super(Preprocessing, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride

        self.pe = PositionalEmbedding(seq_len)
        self.pae_x = PatchEmbedding(d_model, patch_len, stride, padding, dropout, dims)
        self.pae_air = PatchEmbedding(d_model, patch_len, stride, padding, dropout, dims)
        self.pae_static = PatchEmbedding(d_model, patch_len, stride, padding, dropout, dims)
        self.pae_time = PatchEmbedding(d_model, patch_len, stride, padding, dropout, dims)

    def forward(self, x, time_stamp, air=None, static=None):
        # 0.
        bs = x.shape[0]
        self.pe = self.pe.to(x.device)
        self.pae_x = self.pae_x.to(x.device)
        self.pae_air = self.pae_air.to(x.device)
        self.pae_static = self.pae_static.to(x.device)
        self.pae_time = self.pae_time.to(x.device)

        # 1. Turn static factors to time series
        if static is not None:
            static = static.unsqueeze(1).repeat(1, self.seq_len, 1)   # [bs, seq_len, static_vars]
            static_pe = self.pe(static.transpose(1, 2))
            static_pe = static_pe.transpose(1, 2)  # [1, seq_len, static_vars]
            static = static * static_pe.clone()  # [bs, seq_len, static_vars]

        # 2. Patch all time series
        x = self.pae_x(x.clone() if x.requires_grad else x)  # [bs*n_vars, n_patches, d_model]
        _, npa, d = x.shape
        x = x.reshape(bs, -1, npa, d)
        if air is not None:
            air = self.pae_air(air)  # [bs*air_vars, n_patches, d_model]
            air = air.reshape(bs, -1, npa, d)
        if static is not None:
            static = self.pae_static(static)  # [bs*static_vars, n_patches, d_model]
            static = static.reshape(bs, -1, npa, d)

        time_stamp = self.pae_time(time_stamp)
        time_stamp = time_stamp.reshape(bs, -1, npa, d)

        return x, time_stamp, air, static


class ModuleSelector(nn.Module):
    def __init__(self):
        super(ModuleSelector, self).__init__()

    def forward(self, x, l):
        """
        :param x: pollutant_data with shape [bs*n_vars, n_patches, d_model]
        :param l: basic represent matrix with shape [n_vars, n_vars]
        :return: a new represent matrix with shape [bs*n_vars, n_vars]
        """
        bs = int(x.shape[0] // l.shape[0])
        l_expand = l.unsqueeze(0).repeat(bs, 1, 1).reshape(x.shape[0], -1)  # [bs*n_vars, n_vars]

        x_pool = x.mean(dim=1)
        x_norm = f.normalize(x_pool, dim=-1)  # [bs*n_vars, d_model]
        sim = x_norm @ x_norm.T  # [bs*n_vars, bs*n_vars]
        sim = f.softmax(sim, dim=-1)

        new_l = sim @ l_expand

        return new_l


class MultiModalAttention(nn.Module):
    def __init__(self,
                 n_head,
                 attn_type,
                 mask_flag,
                 scale,
                 tau,
                 delta,
                 attention_dropout=0.1,
                 output_attention=False,
                 last_dim=1,
                 d_model=1,
                 init_model="kaiming",
                 no_air=False,
                 no_static=False
    ):
        super(MultiModalAttention, self).__init__()
        self.self_attn = SelfAttentionLayer(n_head, attn_type, mask_flag, scale, tau, delta, attention_dropout,
                                            output_attention, last_dim, d_model, init_model)
        self.air_attn = CrossAttentionLayer(n_head, scale, output_attention, d_model, init_model)
        self.static_attn = CrossAttentionLayer(n_head, scale, output_attention, d_model, init_model)
        self.time_attn = CrossAttentionLayer(n_head, scale, output_attention, d_model, init_model)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.25 if not no_air else 0.0))
        self.gamma = nn.Parameter(torch.tensor(0.15 if not no_static else 0.0))
        self.delta = nn.Parameter(torch.tensor(0.1))
        self.output_attention = output_attention

    def forward(self, x, stamp, air=None, static=None):
        self_output, self_weight = self.self_attn(x)
        time_output, time_weight = self.time_attn(x, stamp, stamp)

        _all = self.alpha * self_output + self.gamma * time_output

        if air is None:
            with torch.no_grad():
                self.beta = nn.Parameter(torch.tensor(0.0))  # 重新赋值

        if static is None:
            with torch.no_grad():
                self.gamma = nn.Parameter(torch.tensor(0.0))  # 重新赋值

        if air is not None:
            air_output, air_weight = self.air_attn(x, air, air)
            _all += self.beta * air_output
        else:
            air_weight = None

        if static is not None:
            static_output, static_weight = self.static_attn(x, static, static)
            _all += self.gamma * static_output
        else:
            static_weight = None

        if self.output_attention:
            return _all, self_weight, time_weight, air_weight, static_weight
        else:
            return _all



class TriBlock(nn.Module):
    def __init__(self,
                 n_layers,
                 n_pollutants,
                 pollutant_encoding,
                 n_heads=1,
                 attn_type="DS",
                 mask_flag=True,
                 scale=None,
                 tau=1.,
                 delta=0.,
                 attention_dropout=0.1,
                 output_attention=False,
                 last_dim=512,
                 d_model=512,
                 init_model="kaiming",
                 no_air=False,
                 no_static=False
                 ):
        super(TriBlock, self).__init__()
        self.ModSelector = ModuleSelector()
        self.MMAList = nn.ModuleList([
                nn.ModuleList([
                    nn.Sequential(
                        nn.LayerNorm(d_model),
                        nn.LayerNorm(d_model),
                        nn.LayerNorm(d_model),
                        nn.LayerNorm(d_model),
                        MultiModalAttention(
                            n_heads, attn_type, mask_flag, scale, tau, delta, attention_dropout,
                            output_attention, last_dim, d_model, init_model, no_air, no_static
                        )
                    ) for _ in range(n_layers // n_pollutants)
            ])
            for _ in range(n_pollutants)
        ])
        self.mma_encoding = pollutant_encoding  # pollutant encoding for MMAList
        self.output_attention = output_attention

    def forward(self, x, time_stamp, air, static):
        pollutant_num = x.size(1)
        output = []
        for i in range(pollutant_num):
            pollutant_x = x[:, i].unsqueeze(1)
            mod = self.MMAList[i]
            for m in mod:
                if self.output_attention:
                    pollutant_x, time_stamp= m[0](pollutant_x), m[1](time_stamp)
                    air = m[2](air) if air is not None else None
                    static = m[3](static) if static is not None else None
                    pollutant_x, self_attn, time_attn, air_attn, static_attn = m[4](pollutant_x, time_stamp, air, static)
                else:
                    pollutant_x, time_stamp = m[0](pollutant_x), m[1](time_stamp)
                    air = m[2](air) if air is not None else None
                    static = m[3](static) if static is not None else None
                    pollutant_x = m[4](pollutant_x, time_stamp, air, static)
            output.append(pollutant_x)
        output = torch.cat(output, dim=1)
        return output


class TriPFormer(nn.Module):
    def __init__(self,
                 normalize=False,
                 n_blocks=3,
                 seq_len=336,
                 pred_len=168,
                 d_model=512,
                 patch_len=24,
                 stride=12,
                 padding=0,
                 pre_dropout=0.1,
                 dims=(1, 2),
                 n_layers=3,
                 n_pollutants=3,
                 n_heads=4,
                 attn_type="DM",
                 mask_flag=False,
                 scale=None,
                 tau=None,
                 delta=0.,
                 attention_dropout=0.1,
                 output_attention=False,
                 init_model="kaiming",
                 no_air=False,
                 no_static=False
    ):
        super(TriPFormer, self).__init__()
        self.normalize = normalize
        self.pe = PollutantEncoding().encoding
        self.pre = Preprocessing(seq_len=seq_len, d_model=d_model, patch_len=patch_len, stride=stride, padding=padding,
                                 dropout=pre_dropout, dims=dims)
        self.tri = nn.ModuleList([
            TriBlock(n_layers=n_layers, n_pollutants=n_pollutants, pollutant_encoding=self.pe, n_heads=n_heads,
                     attn_type=attn_type, mask_flag=mask_flag, scale=scale, tau=tau, delta=delta,
                     attention_dropout=attention_dropout, output_attention=output_attention, last_dim=d_model,
                     d_model=d_model, init_model=init_model, no_air=no_air, no_static=no_static)
            for _ in range(n_blocks)
        ])
        self.output_projection = nn.Linear(d_model, pred_len)

    def forward(self, x, time_stamp, air=None, static=None):
        # 0. Normalization
        for k, v in self.pe.items():
            v = torch.tensor(v).to(x.device)
        self.pre = self.pre.to(x.device)
        if self.normalize:
            x_means = x.mean(dim=1, keepdim=True).detach()
            air_means = air.mean(dim=1, keepdim=True).detach() if air is not None else None
            x_stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            air_stds = torch.sqrt(torch.var(air, dim=1, keepdim=True, unbiased=False) + 1e-5) if air is not None else None

            x = (x - x_means) / x_stds
            air = (air - air_means) / air_stds if air else None

            x_means, x_stds = x_means.transpose(1, 2), x_stds.transpose(1, 2)

        # 1. Preprocessing
        x, time_stamp, air, static = self.pre(x, time_stamp, air, static)

        # 2. TriBlock
        for mod in self.tri:
            output = mod(x, time_stamp, air, static)
            x = x + output

        # 3. output_projection
        x = x.mean(dim=2)
        x = self.output_projection(x)

        if self.normalize:
            x = x * x_stds + x_means

        return x

    def pack_forward(self, input):
        x, _, _, time_stamp, _, air, _, static = input
        prediction = self.forward(x, time_stamp, air, static)
        return prediction


# Wrapper class
class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.model = TriPFormer(
            normalize=configs.normalize,
            n_blocks=configs.n_blocks,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            patch_len=configs.patch_len,
            stride=configs.stride,
            padding=configs.padding,
            pre_dropout=configs.pre_dropout,
            dims=configs.dims,
            n_layers=configs.n_layers,
            n_pollutants=configs.n_pollutants,
            n_heads=configs.n_heads,
            attn_type=configs.attn_type,
            mask_flag=configs.mask_flag,
            scale=configs.scale,
            tau=configs.tau,
            delta=configs.delta,
            attention_dropout=configs.attention_dropout,
            output_attention=configs.output_attention,
            init_model=configs.init_model,
            no_air=configs.no_air,
            no_static=configs.no_static
        )

    def forward(self, input):
        return self.model.pack_forward(input)
