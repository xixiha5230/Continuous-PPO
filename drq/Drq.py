import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os.path as path

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from utils.ConfigHelper import ConfigHelper


class RandomShiftsAug(nn.Module):
    def __init__(self, config: ConfigHelper):
        super().__init__()
        self.conf = config
        self.drq_image_pad = config.drq_image_pad
        self.drq_ray_pad = config.drq_ray_pad

    def forward(self, x):
        if not self.conf.use_drq:
            return x
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2)
            n, c, h, w = x.size()
            assert h == w
            padding = tuple([self.drq_image_pad] * 4)
            x = F.pad(x, padding, "replicate")
            eps = 1.0 / (h + 2 * self.drq_image_pad)
            arange = torch.linspace(
                -1.0 + eps,
                1.0 - eps,
                h + 2 * self.drq_image_pad,
                device=x.device,
                dtype=x.dtype,
            )[:h]
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

            shift = torch.randint(
                0,
                2 * self.drq_image_pad + 1,
                size=(n, 1, 1, 2),
                device=x.device,
                dtype=x.dtype,
            )
            shift *= 2.0 / (h + 2 * self.drq_image_pad)

            grid = base_grid + shift
            x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
            x = x.permute(0, 2, 3, 1)
            return x
        elif len(x.shape) == 2 and x.shape[-1] == 802:
            # 对向量进行padding
            padding = tuple([self.drq_ray_pad] * 2)
            vec_padded = F.pad(x, padding, "replicate")

            # 生成随机平移的大小
            shift = torch.randint(
                low=-self.drq_ray_pad, high=self.drq_ray_pad + 1, size=(1,)
            )

            # 进行随机平移
            vec_shifted = torch.roll(vec_padded, shifts=shift.item(), dims=1)

            # 去掉padding
            vec_shifted = vec_shifted[:, self.drq_ray_pad : -self.drq_ray_pad]
            return vec_shifted
        else:
            return x


if __name__ == "__main__":
    c = ConfigHelper("configs/UGVRace.yaml")
    c.use_drq = True
    c.drq_image_pad = 4
    r = RandomShiftsAug(c)

    x = torch.arange(0, 802).unsqueeze(0).to(c.device).float()
    print(x.shape)
    nx = r(x)
    print(nx.shape)
