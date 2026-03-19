import torch
import torch.nn as nn

# hand_category(17) + blocker_flags(4) + board_texture(4) + opp_range(13) = 38
FEAT_DIM = 38


class DiscardNet(nn.Module):
    def __init__(self, feat_dim: int = FEAT_DIM, hidden: int = 128, n_layers: int = 3):
        super().__init__()
        layers = []
        in_dim = feat_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden), nn.ReLU()]
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
