import math

import torch


class SignSGDCompressor:
    def __init__(self, local_lr: float, server_lr: float, device) -> None:
        self.device = device
        self.local_lr = local_lr
        self.server_lr = server_lr

    def compress(
        self,
        updates: torch.Tensor,
    ):
        compressed_delta = []
        num_params = 0
        num_ones = 0
        for _i, (_name, param) in enumerate(updates.items()):
            signed_param = torch.sign(param)
            compressed_delta.append(signed_param.cpu().numpy())
            # Add estimated ps
            num_param = torch.numel(signed_param)
            num_params += num_param
            num_ones += (torch.sum(signed_param) + num_param) / 2
        local_freq = num_ones / num_params
        local_bitrate = self.find_bitrate(
            [local_freq + 1e-50, 1 - local_freq + 1e-50], num_params
        )

        return compressed_delta, (local_bitrate / num_params).cpu().numpy()

    def find_bitrate(self, probs, num_params):
        local_bitrate = 0
        for p in probs:
            local_bitrate += p * math.log2(1 / p)
        return local_bitrate * num_params
