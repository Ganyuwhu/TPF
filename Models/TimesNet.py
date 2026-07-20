import argparse

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.fft
from Layers.Embed import DataEmbedding, StaticTimeEmbedding
from Layers.ConvBlocks import InceptionBlockV1


def fft_for_period(x, k=2):
    # 只在需要时计算，使用float32
    xf = torch.fft.rfft(x.float(), dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    # 避免CPU传输，在GPU上计算周期
    period = (x.shape[1] // top_list).cpu().numpy()
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlockV1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlockV1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = fft_for_period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = f.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """
    def __init__(self, task_name, seq_len, label_len, pred_len, top_k, enc_in, d_model, d_ff, num_kernels, embed, freq,
                 dropout, e_layers, target, c_out, num_class, static_dim):
        super(TimesNet, self).__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.model = nn.ModuleList([TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels) for _ in range(e_layers)])
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.target = target
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(d_model, c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(d_model, c_out, bias=True)
        if self.task_name == 'classification':
            self.act = f.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * seq_len, num_class)
        self.static_dim = static_dim

        if self.static_dim is not None:
            self.static_embedding = StaticTimeEmbedding(static_dim)
            self.static_linear = nn.Linear(static_dim, enc_in)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        output = dec_out[:, -self.pred_len:]
        if output.shape[-1] == len(self.target):
            output = output.permute(0, 2, 1)
        return output

    def imputation(self, x_enc, x_mark_enc, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def anomaly_detection(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out.mul(
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        dec_out = dec_out.add(
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1)))
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, input_datas, mask=None):
        x_enc, label, y, x_time_stamp, label_time_stamp, air, air_label, static = input_datas
        if self.static_dim is not None:
            static_embed = self.static_embedding(static, x_time_stamp)
            static_embed = f.relu(self.static_linear(static_embed))
            x_enc = x_enc + static_embed

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_time_stamp)
            dec_out = dec_out[:, -self.pred_len:, :].permute(0, 2, 1)  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_time_stamp, label)
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_time_stamp)
        else:
            dec_out = self.forecast(x_enc, x_time_stamp)

        return dec_out


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = TimesNet(
            task_name=configs.task_name,
            seq_len=configs.seq_len,
            label_len=configs.label_len,
            pred_len=configs.pred_len,
            top_k=configs.top_k,
            enc_in=configs.enc_in,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            num_kernels=configs.num_kernels,
            embed=configs.embed,
            freq=configs.freq,
            dropout=configs.dropout,
            e_layers=configs.e_layers,
            target=configs.target,
            c_out=configs.c_out,
            num_class=configs.num_class,
            static_dim=configs.static_dim
        )

    def forward(self, input):
        return self.model(input)


if __name__ == "__main__":
    timesnet = TimesNet(
        task_name='long_term_forecast',
        seq_len=336,
        label_len=168,
        pred_len=168,
        top_k=5,
        enc_in=3,
        d_model=64,
        d_ff=128,
        num_kernels=6,
        embed='fixed',
        freq='h',
        dropout=0.1,
        e_layers=3,
        target=['NO2', 'PM2.5', 'O3'],
        c_out=3,
        num_class=3,
        static_dim=None
    )

    x = torch.rand((64, 336, 3))
    label = torch.rand((64, 168, 3))
    y = torch.rand((64, 168, 3))
    x_time_stamp = torch.rand((64, 336, 4))
    label_time_stamp = torch.rand((64, 168, 4))
    air = torch.rand((64, 168, 3))
    air_label = torch.rand((64, 168, 3))
    static = torch.rand((64, 29))
    input_datas = x, label, y, x_time_stamp, label_time_stamp, air, air_label, static

    y = timesnet(input_datas)
    print(y.shape)
