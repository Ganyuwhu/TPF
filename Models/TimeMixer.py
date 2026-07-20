import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._higher_order_ops import cond
import torch.nn.functional as f

from Layers.Embed import DataEmbeddingWoPos, StaticTimeEmbedding
from Layers.StandardNorm import Normalize

from Layers.Average import MovingAvg, SeriesDecomp


class DftSeriesDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k: int = 5):
        super(DftSeriesDecomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, k=self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf, dim=1)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern
    """

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleSeasonMixing, self).__init__()

        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),

                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):

        # mixing high->low
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern
    """

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super(MultiScaleTrendMixing, self).__init__()

        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ])

    def forward(self, trend_list):

        # mixing low->high
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    def __init__(self, seq_len, pred_len, down_sampling_window, down_sampling_layers, d_model, dropout,
                 channel_independence, decomp_method, top_k, d_ff, kernel_size):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence

        if decomp_method == 'moving_avg':
            self.decompsition = SeriesDecomp(kernel_size)
        elif decomp_method == "dft_decomp":
            self.decompsition = DftSeriesDecomp(top_k)
        else:
            raise ValueError('decompsition is error')

        if not channel_independence:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

        # Mixing season
        self.mixing_multi_scale_season = MultiScaleSeasonMixing(seq_len, down_sampling_window, down_sampling_layers)

        # Mxing trend
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(seq_len, down_sampling_window, down_sampling_layers)

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if not self.channel_independence:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        # bottom-up season mixing
        out_season_list = self.mixing_multi_scale_season(season_list)
        # top-down trend mixing
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(x_list, out_season_list, out_trend_list,
                                                      length_list):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class TimeMixer(nn.Module):

    def __init__(self, task_name, seq_len, label_len, pred_len, down_sampling_window, down_sampling_method, channel_independence, e_layers,
                 enc_in, c_out,target, d_model, embed, freq, dropout, use_norm, down_sampling_layers, num_class, decomp_method,
                 top_k, d_ff, kernel_size, static_dim):
        super(TimeMixer, self).__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_method = down_sampling_method
        self.down_sampling_layers = down_sampling_layers
        self.channel_independence = channel_independence
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(seq_len, pred_len, down_sampling_window, down_sampling_layers,
                                                                d_model, dropout, channel_independence, decomp_method,
                                                                top_k, d_ff, kernel_size)
                                         for _ in range(e_layers)])

        self.preprocess = SeriesDecomp(kernel_size)
        self.enc_in = enc_in
        self.target = target
        self.c_out = c_out
        self.static_dim = static_dim

        if self.static_dim is not None:
            self.static_embedding = StaticTimeEmbedding(static_dim)
            self.static_linear = nn.Linear(static_dim, enc_in)

        if self.channel_independence:
            self.enc_embedding = DataEmbeddingWoPos(1, d_model, embed, freq, dropout)
        else:
            self.enc_embedding = DataEmbeddingWoPos(enc_in, d_model, embed, freq, dropout)

        self.layer = e_layers

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(enc_in, affine=True, non_norm=True if use_norm == 0 else False)
                for i in range(down_sampling_layers + 1)
            ]
        )

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_layers = torch.nn.ModuleList(
                [
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i), pred_len
                    )
                    for i in range(down_sampling_layers + 1)
                ]
            )

            if self.channel_independence:
                self.projection_layer = nn.Linear(d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(d_model, c_out, bias=True)

                self.out_res_layers = torch.nn.ModuleList([
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    )
                    for i in range(down_sampling_layers + 1)
                ])

                self.regression_layers = torch.nn.ModuleList(
                    [
                        torch.nn.Linear(
                            seq_len // (down_sampling_window ** i),
                            pred_len,
                        )
                        for i in range(down_sampling_layers + 1)
                    ]
                )

        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            if self.channel_independence:
                self.projection_layer = nn.Linear(d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(d_model, c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * seq_len, num_class)

    def out_projection(self, dec_out, i, out_res):
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        if self.channel_independence:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.enc_in, out_channels=self.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        x_mark_enc = x_mark_sampling_list if x_mark_enc is not None else None

        return x_enc, x_mark_enc

    def forecast(self, input_datas):
        x, label, y, x_time_stamp, label_time_stamp, air, air_label, static = input_datas
        if self.static_dim is not None:
            static_embed = self.static_embedding(static, x_time_stamp)
            static_embed = f.relu(self.static_linear(static_embed))
            x = x + static_embed

        x, x_time_stamp = self.__multi_scale_process_inputs(x, x_time_stamp)

        x_list = []
        x_mark_list = []
        if x_time_stamp is not None:
            for i, x, x_mark in zip(range(len(x)), x, x_time_stamp):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_list.append(x)
                    x_mark = x_mark.repeat(N, 1, 1)
                    x_mark_list.append(x_mark)
                else:
                    x_list.append(x)
                    x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x)), x, ):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_time_stamp is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)  # [B,T,C]
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)  # [B,T,C]
                enc_out_list.append(enc_out)

        # Past Decomposable Mixing as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # Future Multipredictor Mixing as decoder for future
        dec_out_list = self.future_multi_mixing(B, enc_out_list, x_list)

        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        if dec_out.shape[-1] == len(self.target):
            dec_out = dec_out.permute(0, 2, 1)
        return dec_out

    def future_multi_mixing(self, B, enc_out_list, x_list):
        dec_out_list = []
        if self.channel_independence:
            x_list = x_list[0]
            for i, enc_out in zip(range(len(x_list)), enc_out_list):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, enc_out, out_res in zip(range(len(x_list[0])), enc_out_list, x_list[1]):
                dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(
                    0, 2, 1)  # align temporal dimension
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def classification(self, x_enc, x_mark_enc):
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)
        x_list = x_enc

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        enc_out = enc_out_list[0]
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

    def anomaly_detection(self, x_enc):
        B, T, N = x_enc.size()
        x_enc, _ = self.__multi_scale_process_inputs(x_enc, None)

        x_list = []

        for i, x in zip(range(len(x_enc)), x_enc, ):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def imputation(self, x_enc, x_mark_enc, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        B, T, N = x_enc.size()
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)
                x_mark = x_mark.repeat(N, 1, 1)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc, ):
                B, T, N = x.size()
                if self.channel_independence:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # embedding
        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B,T,C]
            enc_out_list.append(enc_out)

        # MultiScale-CrissCrossAttention  as encoder for past
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        dec_out = self.projection_layer(enc_out_list[0])
        dec_out = dec_out.reshape(B, self.configs.c_out, -1).permute(0, 2, 1).contiguous()

        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        return dec_out

    def forward(self, input_datas):
        dec_out = self.forecast(input_datas)
        return dec_out


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.model = TimeMixer(task_name=config.task_name,
                               seq_len=config.seq_len,
                               label_len=config.label_len,
                               pred_len=config.pred_len,
                               down_sampling_window=config.down_sampling_window,
                               down_sampling_method=config.down_sampling_method,
                               channel_independence=config.channel_independence,
                               e_layers=config.e_layers,
                               enc_in=config.enc_in,
                               c_out=config.c_out,
                               target=config.target,
                               d_model=config.d_model,
                               embed=config.embed,
                               freq=config.freq,
                               dropout=config.dropout,
                               use_norm=config.use_norm,
                               down_sampling_layers=config.down_sampling_layers,
                               num_class=config.num_class,
                               decomp_method=config.decomp_method,
                               top_k=config.top_k,
                               d_ff=config.d_ff,
                               kernel_size=config.kernel_size,
                               static_dim=config.static_dim
        )

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    timemixer = TimeMixer(
        task_name='long_term_forecast',
        seq_len=336,
        label_len=168,
        pred_len=168,
        down_sampling_window=2,
        down_sampling_method='avg',
        channel_independence=True,
        e_layers=3,
        enc_in=3,
        c_out=3,
        target=['NO2', 'PM2.5', 'O3'],
        d_model=64,
        embed='fixed',
        freq='h',
        dropout=0.1,
        use_norm=1,
        down_sampling_layers=3,
        num_class=3,
        decomp_method='moving_avg',
        top_k=5,
        d_ff=128,
        kernel_size=25,
        static_dim=28
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

    y = timemixer(input_datas)
    print(y.shape)
