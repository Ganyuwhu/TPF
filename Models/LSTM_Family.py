import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as f
from matplotlib import pyplot as plt

from statsmodels.tsa.seasonal import STL
from Layers.Embed import *


class LSTM_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, bidirectional=False):
        super().__init__()

        self.lstm = nn.LSTM(
            num_layers=num_layers,
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.context_window = seq_len
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_layers = num_layers
        self.rnn_directions = 2 if bidirectional else 1

    def forward(self, input_data):
        ht = torch.zeros(self.num_layers * self.rnn_directions, input_data.size(0), self.hidden_size, device=input_data.device)
        ct = ht.clone()
        if input_data.ndim < 3:
            input_data.unsqueeze(-1)
        lstm_out, (ht, ct) = self.lstm(input_data, (ht, ct))
        if self.rnn_directions > 1:
            lstm_out = lstm_out.contiguous().reshape(input_data.size(0), self.context_window, self.rnn_directions, self.hidden_size)
            lstm_out = torch.sum(lstm_out, dim=2)
        return lstm_out, (ht, ct)


class AttentionDecoderCell(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size):
        super().__init__()

        self.attention_linear = nn.Linear(hidden_size + input_size, seq_len)
        self.decoder_cell = nn.LSTMCell(
            input_size=hidden_size,
            hidden_size=hidden_size
        )
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, encoder_output, prev_hidden, prev_cell, y):
        if prev_hidden.ndimension() == 3:
            prev_hidden = prev_hidden[-1]  # 保存最后一层的信息
        if prev_cell.ndimension() == 3:
            prev_cell = prev_cell[-1]
        attention_input = torch.cat((prev_hidden, y), dim=1)
        attention_weights = torch.nn.functional.softmax(self.attention_linear(attention_input), dim=-1).unsqueeze(1)
        attention_combine = torch.bmm(attention_weights, encoder_output).squeeze(1)
        h_next, c_next = self.decoder_cell(attention_combine, (prev_hidden, prev_cell))

        output = self.out(h_next)

        return output, (h_next, c_next)


class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, seq_len, pred_len, target, bidirectional=False, teacher_forcing=0.3):
        super().__init__()

        self.encoder = LSTM_Encoder(input_size, hidden_size, num_layers, seq_len, bidirectional)
        self.decoder_cell = AttentionDecoderCell(input_size, seq_len, hidden_size)
        self.output_size = output_size
        self.input_size = input_size
        self.target_window = pred_len
        self.target = target
        self.teacher_forcing = teacher_forcing
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_data, y=None):
        x = input_data[f'{self.target[0]}_x']
        if x.ndimension() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1).contiguous()
        encoder_output, (encoder_hidden, encoder_cell) = self.encoder(x)
        prev_hidden, prev_cell = encoder_hidden, encoder_cell
        predict = torch.zeros(self.target_window, x.size(0), self.input_size).to(x.device)
        y_prev = x[:, -1, :]
        for t in range(self.target_window):
            rnn_output, (prev_hidden, prev_cell) = self.decoder_cell(encoder_output, prev_hidden, prev_cell, y_prev)
            y_prev = rnn_output
            predict[t] = rnn_output
        output = {
            'predict': self.linear(predict).permute(1, 2, 0)
        }
        return output


######################################
# TCN-BiLSTM-DMAttention with STL
class CDConv(nn.Module):
    # Casual Dilated Conv
    def __init__(self, input_size, output_size, kernel_size, dilation=1, stride=1, padding=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding

        self.conv1d = nn.Conv1d(
            in_channels=input_size,
            out_channels=output_size,
            kernel_size=kernel_size,
            dilation=dilation,
            stride=stride,
            padding=padding
        )

    def forward(self, input_data):
        pad_len = (self.kernel_size - 1) * self.dilation
        input_datas = f.pad(input_data, (pad_len, 0))
        output = f.relu(self.conv1d(input_datas))
        return output


class TCNResidualCDConv(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dilation=(1, 2, 4), stride=1, padding=0, num_layers=3):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.num_layers = 3

        self.residual_layer = nn.Conv1d(
            in_channels=input_size[0],
            out_channels=output_size[-1],
            kernel_size=1
        )

        cd_conv_layers = nn.ModuleList()
        for i in range(num_layers):
            cd_conv_layers.append(
                CDConv(
                    input_size=input_size[i],
                    output_size=output_size[i],
                    kernel_size=kernel_size[i],
                    dilation=dilation[i]
                )
            )
        self.cd_conv_layers = nn.Sequential(*cd_conv_layers)

    def forward(self, input_data):
        residual = self.residual_layer(input_data)
        cd_conv_output = self.cd_conv_layers(input_data)
        return residual + cd_conv_output


class DMAttention(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.tensor(gamma), requires_grad=True)

    def forward(self, input_datas):
        # 1. dot
        a_T = input_datas.permute(0, 2, 1)
        a = input_datas
        b = f.softmax(torch.bmm(a_T, a), dim=1)
        return self.gamma * (torch.bmm(a, b)) + a


class TCN_BiLSTM_DMAttention(nn.Module):
    def __init__(self, input_size, output_size, seq_len, pred_len, kernel_size, dilation=(1, 2, 4), stride=1, padding=0, conv_layers=3,
                 hidden_size=4, lstm_layers=2, gamma=1.):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        self.padding = padding
        self.conv_layers = conv_layers
        self.hidden_dim = hidden_size
        self.lstm_layers = lstm_layers
        self.gamma = gamma

        self.tcn_layer = TCNResidualCDConv(
            input_size=input_size,
            output_size=output_size,
            kernel_size=kernel_size,
            dilation=dilation,
            num_layers=conv_layers
        )

        self.BiLstm = nn.LSTM(
            input_size=self.output_size[-1],
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True
        )

        self.fc_1 = nn.Linear(2 * hidden_size, input_size[0])

        self.dma = DMAttention(gamma)

        self.fc_2 = nn.Linear(seq_len, pred_len)

    def forward(self, input_datas):
        tcn_output = self.tcn_layer(input_datas).permute(0, 2, 1).contiguous()
        bi_lstm_output, _ = self.BiLstm(tcn_output)
        bi_lstm_output = self.fc_1(bi_lstm_output).permute(0, 2, 1)
        dma_output = self.dma(bi_lstm_output)
        output = self.fc_2(dma_output)
        return output


class multi_TCN_BiLSTM_DMAttention(nn.Module):
    def __init__(self, input_size, output_size, seq_len, pred_len, kernel_size, target, dilation=(1, 2, 4), conv_stride=1, padding=0, conv_layers=3,
                 hidden_size=4, lstm_layers=2, gamma=1., static_dim=None, enc_in=3):
        super().__init__()

        self.input_size= input_size
        self.output_size = output_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.kernel_size = kernel_size
        self.target = target
        self.dilation = dilation
        self.conv_stride = conv_stride
        self.padding = padding
        self.conv_layers = conv_layers
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.gamma = gamma
        self.static_dim = static_dim
        self.enc_in = enc_in

        if self.static_dim is not None:
            self.static_embedding = StaticTimeEmbedding(static_dim)
            self.static_linear = nn.Linear(static_dim, enc_in)

        self.module_list = nn.ModuleList()
        for pollutant in target:
            self.module_list.append(TCN_BiLSTM_DMAttention(input_size, output_size, seq_len, pred_len, kernel_size, dilation,
                                    conv_stride, padding, conv_layers, hidden_size, lstm_layers, gamma))

    def forward(self, input_datas):
        x, label, y, x_time_stamp, label_time_stamp, air, air_label, static = input_datas
        if self.static_dim is not None:
            static_embed = self.static_embedding(static, x_time_stamp)
            static_embed = f.relu(self.static_linear(static_embed))
            x = x + static_embed
        predict = []
        for i, pollutant in enumerate(self.target):
            data = x[:, :, i].unsqueeze(-1).permute(0, 2, 1)
            predict.append(self.module_list[i](data))
        predict = torch.cat(predict, dim=1)
        return predict


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.model = multi_TCN_BiLSTM_DMAttention(
            input_size=configs.input_size,
            output_size=configs.output_size,
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            kernel_size=configs.kernel_size,
            target=configs.target,
            dilation=configs.dilation,
            conv_stride=configs.conv_stride,
            padding=configs.padding,
            conv_layers=configs.conv_layers,
            hidden_size=configs.hidden_size,
            lstm_layers=configs.lstm_layers,
            gamma=configs.gamma,
            static_dim=configs.static_dim,
            enc_in=configs.enc_in
        )

    def forward(self, input_datas, **kwargs):
        return self.model(input_datas)


if __name__ == "__main__":
    lstm = multi_TCN_BiLSTM_DMAttention(
        input_size=(1, 64, 64),
        output_size=(64, 64, 32),
        seq_len=336,
        pred_len=168,
        kernel_size=(3, 5, 3),
        target=['NO2', 'PM2.5', 'O3'],
        dilation=(1, 2, 4),
        conv_stride=1,
        padding=0,
        conv_layers=3,
        hidden_size=4,
        lstm_layers=2,
        gamma=1.,
        static_dim=29,
        enc_in=3
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

    y = lstm(input_datas)
    print(y.shape)
