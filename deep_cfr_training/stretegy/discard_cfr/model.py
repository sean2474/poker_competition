import torch
import torch.nn as nn

# blocker_flags(4) + board_texture(6) + range_features(opp)(17) + range_features(hero)(17) = 44
FEAT_DIM = 44


class _BaseNet(nn.Module):
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


class AdvantageNet(_BaseNet):
    """Predicts instantaneous advantage (regret) for a keep combo.
    Used during CFR traversal to compute the current strategy via regret matching.
    """
    pass


class StrategyNet(_BaseNet):
    """Predicts average strategy weight for a keep combo.
    Trained on accumulated (features, strategy_prob) pairs.
    Used for inference (actual play).
    """
    pass


DiscardNet = AdvantageNet  # backward compat
