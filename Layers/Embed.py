import torch
import torch.nn as nn
import math

from Utils.Tools import Transpose


# Positional Embedding for 2D Tensor with shape [n_vars, seq_len]
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


# Patch Embedding
class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout=0.1, dims=(1, 2), device="cpu"):
        super(PatchEmbedding, self).__init__()
        # 0. transpose
        self.transpose = Transpose(*dims)

        # 1. patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))  # padding 0 dims on the left side and padding dims on the right side

        # 2. input encoding
        self.value_embedding = nn.Linear(self.patch_len, d_model).to(device)

        # 3. positional embedding
        self.pe = PositionalEmbedding(d_model)

    def forward(self, x):
        # 0. transpose
        x = self.transpose(x)  # [bs, n_vars, seq_len]

        # 1. patching
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.contiguous()
        x = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))

        # 2. input encoding and positional embedding
        x = self.value_embedding(x) + self.pe(x).clone()

        return x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class StaticTimeEmbedding(nn.Module):
    def __init__(self, static_dim, time_dim=4, embed_dim=64, embed_type='fixed'):
        """
        Args:
            static_dim: 静态特征维度 (29)
            time_dim: 时间戳维度 (4)
            embed_dim: 时间戳嵌入维度
            embed_type: 嵌入类型 ('fixed' 或 'learnable')
        """
        super(StaticTimeEmbedding, self).__init__()

        self.static_dim = static_dim
        self.time_dim = time_dim
        self.embed_dim = embed_dim

        # 时间戳各维度的类别数
        hour_size = 24  # 0-23
        weekday_size = 7  # 0-6
        day_size = 32  # 1-31
        month_size = 13  # 1-12

        # 选择嵌入类型
        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding

        # 为每个时间特征创建嵌入层
        self.hour_embed = Embed(hour_size, embed_dim)
        self.weekday_embed = Embed(weekday_size, embed_dim)
        self.day_embed = Embed(day_size, embed_dim)
        self.month_embed = Embed(month_size, embed_dim)

        # 将时间嵌入投影到与静态特征相同的维度
        self.time_projection = nn.Linear(embed_dim * 4, static_dim)

        # 可选：添加LayerNorm
        self.layer_norm = nn.LayerNorm(static_dim)

    def forward(self, static, time_stamp):
        """
        Args:
            static: [batch_size, static_dim]
            time_stamp: [batch_size, seq_len, 4]

        Returns:
            [batch_size, seq_len, static_dim]
        """
        batch_size, seq_len, _ = time_stamp.shape

        # 1. 将Static扩充到 [batch_size, seq_len, static_dim]
        # 方法：在seq_len维度上重复
        static_expanded = static.unsqueeze(1).repeat(1, seq_len, 1)  # [64, 336, 29]

        # 2. 对时间戳进行Embedding
        # 转换为长整型
        time_stamp = time_stamp.long()  # [64, 336, 4]

        # 提取各时间特征
        hour = time_stamp[:, :, 0]  # [64, 336]
        weekday = time_stamp[:, :, 1]  # [64, 336]
        day = time_stamp[:, :, 2]  # [64, 336]
        month = time_stamp[:, :, 3]  # [64, 336]

        # 分别嵌入
        hour_emb = self.hour_embed(hour)  # [64, 336, embed_dim]
        weekday_emb = self.weekday_embed(weekday)  # [64, 336, embed_dim]
        day_emb = self.day_embed(day)  # [64, 336, embed_dim]
        month_emb = self.month_embed(month)  # [64, 336, embed_dim]

        # 拼接所有时间嵌入 [64, 336, embed_dim * 4]
        time_emb = torch.cat([hour_emb, weekday_emb, day_emb, month_emb], dim=-1)

        # 投影到静态特征维度 [64, 336, static_dim]
        time_projected = self.time_projection(time_emb)

        # 3. 将Static和Embed进行逐元素乘法
        # [64, 336, static_dim] * [64, 336, static_dim]
        output = static_expanded * time_projected

        # 可选：LayerNorm
        output = self.layer_norm(output)

        return output


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)


class DataEmbeddingWoPos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbeddingWoPos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)

class DataEmbeddingWoPosTemp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbeddingWoPosTemp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x)
        return self.dropout(x)

class DataEmbeddingWoTemp(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbeddingWoTemp, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
