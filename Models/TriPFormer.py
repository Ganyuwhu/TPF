from Layers.Embed import PatchEmbedding, StaticTimeEmbedding
from Layers.AttentionFamily import SelfAttentionLayer, CrossAttentionLayer
from Models.MoHE import *
from timm.layers import DropPath


class Preprocessing(nn.Module):
    def __init__(self, seq_len, d_model, patch_len, stride, static_dim, padding, dropout, dims, time_dim=4,
                 embed_dim=64, embed_type='fixed'):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride
        self.STE = StaticTimeEmbedding(static_dim, time_dim, embed_dim, embed_type)

        self.pae_x = PatchEmbedding(d_model, patch_len, stride, padding, dropout, dims)
        self.pae_air = PatchEmbedding(d_model, patch_len, stride, padding, dropout, dims)
        self.pae_static = PatchEmbedding(d_model, patch_len, stride, padding, dropout, dims)
        self.pae_time = PatchEmbedding(d_model, patch_len, stride, padding, dropout, dims)

    def forward(self, x, time_stamp, air, static):
        bs = x.shape[0]
        # 1. static embedding
        if static is not None:
            static = self.STE(static, time_stamp)

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
    def __init__(self, n_head, attn_type, mask_flag, scale, tau, delta, attention_dropout=0.1, output_attention=False,
                 last_dim=512, d_model=1, n_vars=3, init_model="kaiming", no_air=False, no_static=False, ms_type="self",
                 **kwargs):
        super().__init__()
        self.ms_type = ms_type
        if ms_type == "self":
            self.selector = MultiModalAttention(n_head, attn_type, mask_flag, scale, tau, delta, attention_dropout,
                                                output_attention, last_dim, d_model, n_vars, init_model, no_air,
                                                no_static)
        else:
            pass

    def forward(self, x):
        return self.selector.classification(x)


class MultiModalAttention(nn.Module):
    def __init__(self, n_head, attn_type, mask_flag, scale, tau, delta, attention_dropout=0.1, output_attention=False,
                 last_dim=512, d_model=1, n_vars=3, init_model="kaiming", no_air=False, no_static=False):
        super(MultiModalAttention, self).__init__()
        self.self_attn = SelfAttentionLayer(n_head, attn_type, mask_flag, scale, tau, delta, attention_dropout,
                                            output_attention, last_dim, d_model, init_model)
        self.air_attn = CrossAttentionLayer(n_head, scale, output_attention, d_model, init_model)
        self.static_attn = CrossAttentionLayer(n_head, scale, output_attention, d_model, init_model)
        self.time_attn = CrossAttentionLayer(n_head, scale, output_attention, d_model, init_model)
        self.alpha = nn.Parameter(torch.tensor(0.75))
        self.beta = nn.Parameter(torch.tensor(0.1 if not no_air else 0.0))
        self.gamma = nn.Parameter(torch.tensor(0.1 if not no_static else 0.0))
        self.delta = nn.Parameter(torch.tensor(0.05))
        self.output_attention = output_attention
        self.pool = nn.AdaptiveAvgPool2d((3, d_model))
        self.projection = nn.Sequential(
            nn.Linear(3 * d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_vars),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, stamp=None, air=None, static=None):
        self_output, self_weight = self.self_attn(x)
        _all = self.alpha * self_output

        if stamp is None:
            with torch.no_grad():
                self.gamma = nn.Parameter(torch.tensor(0.0))

        if air is None:
            with torch.no_grad():
                self.beta = nn.Parameter(torch.tensor(0.0))  # 重新赋值

        if static is None:
            with torch.no_grad():
                self.delta = nn.Parameter(torch.tensor(0.0))  # 重新赋值

        if stamp is not None:
            time_output, time_weight = self.time_attn(x, stamp, stamp)
            _all = _all + self.gamma * time_output
        else:
            time_weight = None

        if air is not None:
            air_output, air_weight = self.air_attn(x, air, air)
            _all = _all + self.beta * air_output
        else:
            air_weight = None

        if static is not None:
            static_output, static_weight = self.static_attn(x, static, static)
            _all = _all + self.delta * static_output
        else:
            static_weight = None

        if self.output_attention:
            return _all, self_weight, time_weight, air_weight, static_weight
        else:
            return _all

    def classification(self, x):
        attention_score = self.forward(x, stamp=None, air=None, static=None) if not self.output_attention else self.forward(x, stamp=None, air=None, static=None)[0]
        attention_score = self.pool(attention_score)
        attention_score = attention_score.flatten(start_dim=2)
        return self.projection(attention_score)


class TriBlock(nn.Module):
    def __init__(self, n_heads, n_layers, n_pollutants,  attn_type="DS", mask_flag=True,
                 scale=None, tau=1., delta=0., attention_dropout=0.2, output_attention=False, last_dim=512, d_model=512,
                 d_ff=1024, init_model="kaiming", no_air=False, no_static=False, norm_type='rms', drop_rate=0.5, **kwargs):
        super().__init__()

        self.n_pollutants = n_pollutants
        self.n_layers = n_layers
        layers_per_pollutant = n_layers // n_pollutants
        drop_list = [x.item() for x in torch.linspace(0, drop_rate, self.n_layers)]

        self.MMAList = nn.ModuleList()
        for p_idx in range(n_pollutants):
            pollutant_layers = nn.ModuleList()
            for l_idx in range(layers_per_pollutant):
                # 每一层包含：Norm -> Attention -> DropPath（残差分支）
                layer = nn.ModuleDict({
                    'norm': self.get_norm(norm_type, d_model),
                    'attn': MultiModalAttention(
                        n_heads, attn_type, mask_flag, scale, tau, delta,
                        attention_dropout, output_attention, last_dim, d_model,
                        n_pollutants, init_model, no_air, no_static
                    ),
                    'drop_path': DropPath(drop_list[p_idx * layers_per_pollutant + l_idx])
                                 if drop_list[p_idx * layers_per_pollutant + l_idx] > 0.
                                 else nn.Identity()
                })
                pollutant_layers.append(layer)
            self.MMAList.append(pollutant_layers)

        self.block_norm = self.get_norm(norm_type, d_model)

        self.drop_path_ffn = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, stamp=None, air=None, static=None, classification=None):
        input = x.clone()
        raw_input = x.copy()
        if classification is not None:
            out = []
            for i in range(self.n_pollutants):
                m_list = self.MMAList[i]
                for layer in m_list:
                    _x = layer['norm'](raw_input)
                    attn_out = layer['attn'](_x, stamp=stamp, air=air, static=static, classification=classification)
                    x = x + layer['drop_path'](attn_out)
                out.append(x)

            predict = torch.zeros_like(out[0])
            for i in range(self.n_pollutants):
                weight = classification[:, :, i].unsqueeze(-1).unsqueeze(-1)
                predict = predict + weight * out[i]

            predict = self.ffn(self.block_norm(predict))
            predict = input + self.drop_path_ffn(predict)

        else:
            out = []
            for i in range(self.n_pollutants):
                m_list = self.MMAList[i]
                x1 = x[:, i].unsqueeze(1)
                for layer in m_list:
                    _x = layer['norm'](x1)
                    attn_out = layer['attn'](_x, stamp=stamp, air=air, static=static)
                    x1 = x1 + layer['drop_path'](attn_out)
                out.append(x1)
            predict = torch.cat(out, dim=1)
            predict = self.ffn(self.block_norm(predict))
            predict = x + self.drop_path_ffn(predict)

        return predict

    @ staticmethod
    def get_norm(norm_type, d_model, init_alpha=0.5):
        if norm_type == 'rms':
            return RMSNorm(d_model)
        elif norm_type == 'dyt':
            return DynamicTanh(d_model, init_alpha)
        else:
            return nn.LayerNorm(d_model)


class TriPFormer(nn.Module):
    def __init__(self, seq_len, pred_len, d_model, d_ff, patch_len, stride, static_dim, padding, dropout, dims,
                 time_dim, embed_dim, embed_type, n_heads, attn_type, mask_flag, scale, tau, delta, attention_dropout,
                 output_attention, last_dim, n_vars, init_model, no_air, no_static, norm_type, ms_type, n_layers,
                 drop_rate, separate, **kwargs):
        super().__init__()

        self.separate = separate
        self.PRE = Preprocessing(seq_len, d_model, patch_len, stride, static_dim, padding, dropout, dims, time_dim,
                                 embed_dim, embed_type)
        self.Classifier = ModuleSelector(n_heads, attn_type, mask_flag, scale, tau, delta, attention_dropout,
                                         output_attention, last_dim, d_model, n_vars,
                                         init_model, no_air, no_static, ms_type, **kwargs)
        self.TriBlock = TriBlock(n_heads, n_layers, n_vars,  attn_type, mask_flag, scale, tau, delta, attention_dropout,
                                 output_attention, last_dim, d_model, d_ff, init_model, no_air, no_static, norm_type,
                                 drop_rate)
        self.output_projection = nn.Linear(d_model, pred_len)

    def forward(self, x, time_stamp=None, air=None, static=None):
        x, stamp, air, static = self.PRE(x, time_stamp, air, static)
        classification = self.Classifier(x) if not self.separate else None
        dec_out = self.TriBlock(x, stamp, air, static, classification)
        dec_out = dec_out.mean(dim=2)
        predict = self.output_projection(dec_out)
        return predict

    def pack_forward(self, input_datas):
        x, _, _, time_stamp, _, air, _, static = input_datas
        prediction = self.forward(x, time_stamp, air, static)
        return prediction

    def classify(self, input_datas):
        x, _, _, time_stamp, _, air, _, static = input_datas
        x, stamp, air, static = self.PRE(x, time_stamp, air, static)
        classification = self.Classifier(x)
        return classification


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()

        self.model = TriPFormer(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            d_model=configs.d_model,
            d_ff=configs.d_ff,
            patch_len=configs.patch_len,
            stride=configs.stride,
            static_dim=configs.static_dim,
            padding=configs.padding,
            dropout=configs.dropout,
            dims=configs.dims,
            time_dim=configs.time_dim,
            embed_dim=configs.embed_dim,
            embed_type=configs.embed_type,
            n_heads=configs.n_heads,
            attn_type=configs.attn_type,
            mask_flag=configs.mask_flag,
            scale=configs.scale,
            tau=configs.tau,
            delta=configs.delta,
            attention_dropout=configs.attention_dropout,
            output_attention=configs.output_attention,
            last_dim=configs.last_dim,
            n_vars=configs.n_vars,
            init_model=configs.init_model,
            no_air=configs.no_air,
            no_static=configs.no_static,
            norm_type=configs.norm_type,
            ms_type=configs.ms_type,
            drop_rate=configs.drop_rate,
            n_layers=configs.n_layers,
            separate=configs.separate
        )

    def forward(self, input_datas):
        return self.model.pack_forward(input_datas)

    def classify(self, input_datas):
        return self.model.classify(input_datas)


if __name__ == "__main__":
    model = TriPFormer(
        seq_len=336,
        pred_len=168,
        d_model=512,
        d_ff=1024,
        patch_len=24,
        stride=12,
        static_dim=26,
        padding=0,
        dropout=0.1,
        dims=(1, 2),
        time_dim=4,
        embed_dim=64,
        embed_type='fixed',
        n_heads=4,
        attn_type='DS',
        mask_flag=True,
        scale=None,
        tau=1.,
        delta=0.,
        attention_dropout=0.2,
        output_attention=False,
        last_dim=512,
        n_vars=3,
        init_model=512,
        no_air=False,
        no_static=False,
        ms_type='self',
        norm_type='rms',
        n_layers=9,
        drop_rate=0.5,
        separate=True
    )

    x = torch.rand((64, 336, 3))
    label = torch.rand((64, 168, 3))
    y = torch.rand((64, 168, 3))
    x_time_stamp = torch.rand((64, 336, 4))
    label_time_stamp = torch.rand((64, 168, 4))
    air = torch.rand((64, 336, 3))
    air_label = torch.rand((64, 168, 3))
    static = torch.rand((64, 26))
    input_datas = x, label, y, x_time_stamp, label_time_stamp, air, air_label, static

    y = model(x, x_time_stamp, air, static)
    print(y.shape)

    # for name, param in model.named_parameters():
    #     print(name, '\n')
