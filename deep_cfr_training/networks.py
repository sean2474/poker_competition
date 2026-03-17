"""
Neural networks for Deep CFR.

Two networks:
  - AdvantageNet: predicts advantage values for each action given state
  - One per player (or shared, since symmetric game)

Architecture: 3 hidden layers × 128 neurons, ReLU
Input: 61 features (raw cards + state)
Output: 7 (one per abstract action)
"""

import torch
import torch.nn as nn

from game_env import FEATURE_DIM, NUM_ACTIONS  # FEATURE_DIM = 85


class AdvantageNet(nn.Module):
    """Predicts advantage values: A(s,a) for each action."""
    
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=128, output_dim=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)
    
    def get_strategy(self, features, valid_actions):
        """
        Get action probabilities via regret matching on predicted advantages.
        
        Args:
            features: numpy array of shape (feature_dim,)
            valid_actions: list of valid action indices
        
        Returns:
            strategy: dict of {action: probability}
        """
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            advantages = self.forward(x).squeeze(0).numpy()
        
        # Regret matching on valid actions only
        valid_advs = {a: max(advantages[a], 0) for a in valid_actions}
        total = sum(valid_advs.values())
        
        if total > 0:
            strategy = {a: v / total for a, v in valid_advs.items()}
        else:
            # Paper: "choose the action with highest counterfactual regret with probability 1"
            # This helps RM cope with approximation error (paper Section 2.1)
            best_a = max(valid_actions, key=lambda a: advantages[a])
            strategy = {a: (1.0 if a == best_a else 0.0) for a in valid_actions}
        
        return strategy
    
class StrategyNet(nn.Module):
    """
    Average strategy network: predicts action probabilities directly.
    Trained on (features, strategy) pairs collected during CFR traversals,
    weighted by iteration number (linear averaging).
    
    This is what gets used for actual play — NOT the advantage network.
    """
    
    def __init__(self, input_dim=FEATURE_DIM, hidden_dim=128, output_dim=NUM_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        """Returns raw logits. Apply softmax externally for probabilities."""
        return self.net(x)
    
    def get_action_probs(self, features, valid_actions):
        """Get action probabilities for valid actions only."""
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(x).squeeze(0)
            
            # Mask invalid actions with -inf
            mask = torch.full_like(logits, float('-inf'))
            for a in valid_actions:
                mask[a] = logits[a]
            
            probs = torch.softmax(mask, dim=0).numpy()
        
        return {a: float(probs[a]) for a in valid_actions}
    
