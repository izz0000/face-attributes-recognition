import torch.nn as nn


class LinearBnRelu(nn.Sequential):
    def __init__(self, in_features: int, out_features: int, ) -> None:
        super(LinearBnRelu, self).__init__(
            nn.Linear(in_features, out_features),
            # nn.BatchNorm1d(out_features),
            nn.ReLU()
        )
