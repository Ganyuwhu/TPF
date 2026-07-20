import torch


class TriangleMask:
    def __init__(self, shape, device="cpu"):
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(shape, dtype=torch.bool), diagonal=1).to(device)

    @ property
    def mask(self):
        return self._mask


class ProbMask:
    def __init__(self, shape, index, scores, device="cpu"):
        B, H, L = shape
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)  # 高级索引
        self._mask = indicator.view(scores.shape).to(device)

    @ property
    def mask(self):
        return self._mask
