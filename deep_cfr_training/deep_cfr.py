"""
Deep CFR Training Loop.

Algorithm (Brown et al., 2019):
  1. For each CFR iteration t:
     a. For each player p:
        - Traverse game tree via external sampling
        - At each infoset, compute advantages using current network
        - Store (features, advantages, iteration) in advantage memory
     b. Train advantage network on collected data (weighted by iteration)
  
  2. After all iterations, compute average strategy network from advantage memories.

Key difference from tabular CFR:
  - No abstraction needed — neural network generalizes across similar states
  - Advantage memories are stored as (features, advantage_values) pairs
  - Network is trained to predict advantages from raw features

Usage:
    python deep_cfr.py --iterations 1000 --traversals 1000 --output weights.npz
"""

import argparse
import os
import sys
import time
import random
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from game_env import (
    GameState, deal_game, fast_discard, evaluate_showdown,
    state_to_features, FEATURE_DIM, NUM_ACTIONS,
)
from networks import AdvantageNet, StrategyNet

# GPU device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Training device: {DEVICE}')


class ReservoirBuffer:
    """Fixed-size reservoir sampling buffer for advantage memories."""
    
    def __init__(self, capacity=2_000_000):
        self.capacity = capacity
        self.buffer = []
        self.count = 0
    
    def add(self, features, values, iteration, valid_mask=None):
        self.count += 1
        item = (features, values, iteration, valid_mask)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            idx = random.randint(0, self.count - 1)
            if idx < self.capacity:
                self.buffer[idx] = item
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        features = np.array([b[0] for b in batch])
        values = np.array([b[1] for b in batch])
        iterations = np.array([b[2] for b in batch], dtype=np.float32)
        masks = np.array([b[3] if b[3] is not None else np.ones(values.shape[1]) for b in batch])
        return features, values, iterations, masks
    
    def __len__(self):
        return len(self.buffer)


class DeepCFR:
    """Deep CFR trainer."""
    
    def __init__(self, lr=0.001, buffer_size=2_000_000):
        self.lr = lr
        self.adv_nets = [AdvantageNet() for _ in range(2)]  # stay on CPU for traversal
        self.adv_buffers = [ReservoirBuffer(buffer_size) for _ in range(2)]
        self.optimizers = [optim.Adam(net.parameters(), lr=lr) for net in self.adv_nets]
        
        self.strategy_net = StrategyNet()
        self.strategy_buffer = ReservoirBuffer(buffer_size)
        self.strategy_optimizer = optim.Adam(self.strategy_net.parameters(), lr=lr)
        
        self.iteration = 0
        self.total_iterations = 1
    
    def traverse(self, state, p0_hand, p1_hand, p0_hand5, p1_hand5,
                  community, p0_disc, p1_disc, traversing_player):
        """
        External sampling CFR traversal.
        
        For the traversing player: compute advantages at each decision point.
        For the opponent: sample action from current network strategy.
        
        Returns: expected value for traversing player.
        """
        if state.is_terminal:
            if state.folded_player >= 0:
                # Fold: winner gets loser's bet
                if state.folded_player == traversing_player:
                    return -state.bets[traversing_player]
                else:
                    return state.bets[1 - traversing_player]
            # Showdown
            pot = min(state.bets[0], state.bets[1])
            sd = evaluate_showdown(p0_hand, p1_hand, community)
            if traversing_player == 0:
                return sd * pot
            else:
                return -sd * pot
        
        cp = state.current_player
        valid_actions = state.get_valid_actions()
        
        if len(valid_actions) == 0:
            return 0
        
        # Get features for current player
        if cp == 0:
            hand, hand5, is_bb = p0_hand, p0_hand5, False
            my_disc, opp_disc = p0_disc, p1_disc
        else:
            hand, hand5, is_bb = p1_hand, p1_hand5, True
            my_disc, opp_disc = p1_disc, p0_disc
        
        # Visible community depends on street
        if state.street == 0:
            vis_comm = []
        elif state.street == 1:
            vis_comm = community[:3]
        elif state.street == 2:
            vis_comm = community[:4]
        else:
            vis_comm = community[:5]
        
        features = state_to_features(
            hand, vis_comm,
            state.bets[cp], state.bets[1 - cp],
            state.street, is_bb, my_disc, opp_disc,
            hero_hand5=hand5 if state.street == 0 else None
        )
        
        if cp == traversing_player:
            # Traversing player: compute all action values
            action_values = {}
            for a in valid_actions:
                ns = state.apply(a)
                action_values[a] = self.traverse(
                    ns, p0_hand, p1_hand, p0_hand5, p1_hand5,
                    community, p0_disc, p1_disc, traversing_player
                )
            
            # Compute strategy from advantage network
            strategy = self.adv_nets[cp].get_strategy(features, valid_actions)
            
            # Expected value under current strategy
            ev = sum(strategy[a] * action_values[a] for a in valid_actions)
            
            # Compute advantages
            advantages = np.zeros(NUM_ACTIONS)
            for a in valid_actions:
                advantages[a] = action_values[a] - ev
            
            # Store advantage + valid mask in buffer
            valid_mask = np.zeros(NUM_ACTIONS)
            for a in valid_actions:
                valid_mask[a] = 1.0
            self.adv_buffers[cp].add(features, advantages, self.iteration, valid_mask)
            
            return ev
        
        else:
            # Opponent: sample one action from strategy
            strategy = self.adv_nets[cp].get_strategy(features, valid_actions)
            
            # Store opponent's strategy in strategy memory MΠ (paper Algorithm 2)
            strat_target = np.zeros(NUM_ACTIONS)
            valid_mask = np.zeros(NUM_ACTIONS)
            for a in valid_actions:
                strat_target[a] = strategy[a]
                valid_mask[a] = 1.0
            self.strategy_buffer.add(features, strat_target, self.iteration, valid_mask)
            
            actions = list(strategy.keys())
            probs = [strategy[a] for a in actions]
            chosen = random.choices(actions, weights=probs, k=1)[0]
            
            ns = state.apply(chosen)
            return self.traverse(
                ns, p0_hand, p1_hand, p0_hand5, p1_hand5,
                community, p0_disc, p1_disc, traversing_player
            )
    
    # ── Coroutine-based traverse for batch inference ──────────────────

    def traverse_coro(self, state, p0_hand, p1_hand, p0_hand5, p1_hand5,
                       community, p0_disc, p1_disc, traversing_player):
        """
        Generator version of traverse.
        Yields (features, valid_actions, cp) when network inference is needed.
        Receives strategy dict via .send(strategy).
        Returns EV via StopIteration.value.
        """
        if state.is_terminal:
            if state.folded_player >= 0:
                if state.folded_player == traversing_player:
                    return -float(state.bets[traversing_player])
                else:
                    return float(state.bets[1 - traversing_player])
            pot = min(state.bets[0], state.bets[1])
            sd = evaluate_showdown(p0_hand, p1_hand, community)
            return float(sd * pot if traversing_player == 0 else -sd * pot)

        cp = state.current_player
        valid_actions = state.get_valid_actions()
        if not valid_actions:
            return 0.0

        if cp == 0:
            hand, hand5, is_bb = p0_hand, p0_hand5, False
            my_disc, opp_disc = p0_disc, p1_disc
        else:
            hand, hand5, is_bb = p1_hand, p1_hand5, True
            my_disc, opp_disc = p1_disc, p0_disc

        vis_comm = ([], community[:3], community[:4], community[:5])[min(state.street, 3)]
        features = state_to_features(
            hand, vis_comm, state.bets[cp], state.bets[1 - cp],
            state.street, is_bb, my_disc, opp_disc,
            hero_hand5=hand5 if state.street == 0 else None
        )

        # Yield for batch inference — receive strategy back
        strategy = yield (features, valid_actions, cp)

        if cp == traversing_player:
            action_values = {}
            for a in valid_actions:
                ns = state.apply(a)
                sub = self.traverse_coro(ns, p0_hand, p1_hand, p0_hand5, p1_hand5,
                                          community, p0_disc, p1_disc, traversing_player)
                try:
                    req = next(sub)
                    while True:
                        resp = yield req   # pass inference requests up
                        req = sub.send(resp)
                except StopIteration as e:
                    action_values[a] = e.value

            ev = sum(strategy.get(a, 0) * action_values[a] for a in valid_actions)
            advantages = np.zeros(NUM_ACTIONS)
            valid_mask = np.zeros(NUM_ACTIONS)
            for a in valid_actions:
                advantages[a] = action_values[a] - ev
                valid_mask[a] = 1.0
            self.adv_buffers[cp].add(features, advantages, self.iteration, valid_mask)
            return ev

        else:
            strat_target = np.zeros(NUM_ACTIONS)
            valid_mask = np.zeros(NUM_ACTIONS)
            for a in valid_actions:
                strat_target[a] = strategy.get(a, 0)
                valid_mask[a] = 1.0
            self.strategy_buffer.add(features, strat_target, self.iteration, valid_mask)

            actions = list(strategy.keys())
            probs = [strategy[a] for a in actions]
            chosen = random.choices(actions, weights=probs, k=1)[0]
            ns = state.apply(chosen)
            sub = self.traverse_coro(ns, p0_hand, p1_hand, p0_hand5, p1_hand5,
                                      community, p0_disc, p1_disc, traversing_player)
            try:
                req = next(sub)
                while True:
                    resp = yield req
                    req = sub.send(resp)
            except StopIteration as e:
                return e.value

    @staticmethod
    def _regret_matching(adv_arr, valid_actions):
        """Pure Python regret matching on numpy array."""
        total = 0.0
        best_a, best_v = valid_actions[0], -1e9
        for a in valid_actions:
            v = float(adv_arr[a])
            if v > 0: total += v
            if v > best_v: best_v = v; best_a = a
        if total > 0:
            inv = 1.0 / total
            return {a: max(float(adv_arr[a]), 0) * inv for a in valid_actions}
        return {a: (1.0 if a == best_a else 0.0) for a in valid_actions}

    def run_traversals_batched(self, traversals_per_iter, traversing_player):
        """
        Run `traversals_per_iter` traversals simultaneously.
        At each inference step, batch all waiting generators → single GPU forward.
        """
        # Deal all games at once via C++
        r = __import__('game_env', fromlist=['batch_deal_discard']).batch_deal_discard(traversals_per_iter)
        p0h, p1h, p0d, p1d, comms, p0h5, p1h5 = r

        # Init generators
        gens = {}
        for i in range(traversals_per_iter):
            p0_hand = list(p0h[i])
            p1_hand = list(p1h[i])
            p0_disc = list(p0d[i])
            p1_disc = list(p1d[i])
            comm = list(comms[i])
            p0_5 = list(p0h5[i])
            p1_5 = list(p1h5[i])
            g = self.traverse_coro(GameState(), p0_hand, p1_hand, p0_5, p1_5,
                                    comm, p0_disc, p1_disc, traversing_player)
            gens[i] = g

        # Bootstrap: start all generators
        pending = {}  # idx → (features, valid_actions, cp)
        for i, g in list(gens.items()):
            try:
                req = next(g)
                pending[i] = req
            except StopIteration:
                del gens[i]

        # Run until all done
        while pending:
            # Process each player's batch
            for p in [0, 1]:
                p_idxs = [i for i in list(pending.keys()) if pending[i][2] == p]
                if not p_idxs:
                    continue

                # Batch forward on GPU
                feats = np.stack([pending[i][0] for i in p_idxs])
                x = torch.tensor(feats, dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    adv_batch = self.adv_nets[p](x).cpu().numpy()

                # Resume each generator
                for j, i in enumerate(p_idxs):
                    _, valid_actions, _ = pending.pop(i)
                    strategy = self._regret_matching(adv_batch[j], valid_actions)
                    try:
                        new_req = gens[i].send(strategy)
                        pending[i] = new_req
                    except StopIteration:
                        if i in gens:
                            del gens[i]

    def train_networks(self, batch_size=2048, num_batches=100):
        """Train advantage networks FROM SCRATCH each iteration (paper Section 5.2)."""
        losses = [0, 0]
        for p in range(2):
            if len(self.adv_buffers[p]) < batch_size:
                continue
            
            # CRITICAL: reinitialize network from scratch each iteration (paper Section 5.2)
            self.adv_nets[p] = AdvantageNet().to(DEVICE)
            opt = optim.Adam(self.adv_nets[p].parameters(), lr=self.lr)
            
            net = self.adv_nets[p]
            
            total_loss = 0
            for _ in range(num_batches):
                features, advantages, iterations, masks = self.adv_buffers[p].sample(batch_size)
                
                weights = 2.0 * iterations / max(self.total_iterations, 1)
                
                x = torch.tensor(features, dtype=torch.float32, device=DEVICE)
                y = torch.tensor(advantages, dtype=torch.float32, device=DEVICE)
                w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
                m = torch.tensor(masks, dtype=torch.float32, device=DEVICE)
                
                pred = net(x)
                mask_sum = m.sum() + 1e-8
                loss = ((pred - y) ** 2 * w.unsqueeze(1) * m).sum() / mask_sum
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss += loss.item()
            
            # Move back to CPU for traversal inference
            self.adv_nets[p] = self.adv_nets[p].cpu()
            
            losses[p] = total_loss / num_batches
        
        return losses
    
    def run(self, num_iterations=500, traversals_per_iter=1000,
            train_interval=1, batch_size=2048, num_batches=100):
        """Main Deep CFR training loop."""
        
        self.total_iterations = num_iterations
        
        print(f"Deep CFR Training: {num_iterations} iters × {traversals_per_iter} traversals")
        print(f"Feature dim: {FEATURE_DIM}, Actions: {NUM_ACTIONS}")
        print()
        
        t0 = time.time()
        
        for t in tqdm(range(num_iterations), desc='CFR iters'):
            self.iteration = t + 1
            
            # Batched traversals: both players in one C++ batch deal
            for traversing in range(2):
                self.run_traversals_batched(traversals_per_iter, traversing)
            
            # Train networks periodically
            if (t + 1) % train_interval == 0:
                losses = self.train_networks(batch_size, num_batches)
            
            # Progress
            elapsed = time.time() - t0
            ips = (t + 1) / elapsed
            eta = (num_iterations - t - 1) / ips if ips > 0 else 0
            buf_sizes = [len(b) for b in self.adv_buffers]
            
            if (t + 1) % 10 == 0 or t == 0:
                print(f"  iter {t+1}/{num_iterations}  "
                      f"{ips:.1f} it/s  "
                      f"buffers=[{buf_sizes[0]:,}, {buf_sizes[1]:,}]  "
                      f"loss=[{losses[0]:.4f}, {losses[1]:.4f}]  "
                      f"ETA {int(eta)}s")
        
        # Train average strategy network on strategy buffer
        print("\nTraining average strategy network...")
        self.train_strategy_net(batch_size, num_batches * 3)
        
        elapsed = time.time() - t0
        print(f"\nDone: {num_iterations} iters in {elapsed:.0f}s")
        print(f"Adv buffers: [{len(self.adv_buffers[0]):,}, {len(self.adv_buffers[1]):,}]")
        print(f"Strategy buffer: {len(self.strategy_buffer):,}")
    
    def train_strategy_net(self, batch_size=2048, num_batches=300):
        """Train average strategy network on strategy buffer."""
        if len(self.strategy_buffer) < batch_size:
            print(f"  Strategy buffer too small ({len(self.strategy_buffer)}), skipping")
            return
        
        self.strategy_net = self.strategy_net.to(DEVICE)
        opt = optim.Adam(self.strategy_net.parameters(), lr=self.lr)
        
        total_loss = 0
        for b in range(num_batches):
            features, strategies, iterations, masks = self.strategy_buffer.sample(batch_size)
            
            weights = 2.0 * iterations / max(self.total_iterations, 1)
            
            x = torch.tensor(features, dtype=torch.float32, device=DEVICE)
            y = torch.tensor(strategies, dtype=torch.float32, device=DEVICE)
            w = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
            m = torch.tensor(masks, dtype=torch.float32, device=DEVICE)
            
            logits = self.strategy_net(x)
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs * m * w.unsqueeze(1)).sum() / (m.sum() + 1e-8)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            
            if (b + 1) % 100 == 0:
                print(f"  Strategy net batch {b+1}/{num_batches}, loss={total_loss/(b+1):.4f}")
        
        self.strategy_net = self.strategy_net.cpu()
        print(f"  Strategy net trained: avg loss={total_loss/num_batches:.4f}")
    
    def export(self, path):
        """Export torch models for submission inference."""
        # Strategy net = what gets used for actual play
        torch.save(self.strategy_net.state_dict(), path + '_strategy.pt')
        
        # Full checkpoint for resuming training
        torch.save({
            'strategy_net': self.strategy_net.state_dict(),
            'adv_net_0': self.adv_nets[0].state_dict(),
            'adv_net_1': self.adv_nets[1].state_dict(),
            'iteration': self.iteration,
        }, path + '_full.pt')
        
        total_params = sum(p.numel() for p in self.strategy_net.parameters())
        strategy_size = os.path.getsize(path + '_strategy.pt') / 1024
        print(f"Exported: strategy_net {total_params:,} params, {strategy_size:.0f} KB")
        print(f"Files: {path}_strategy.pt, {path}_full.pt")


def main():
    parser = argparse.ArgumentParser(description="Deep CFR Training")
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--traversals", type=int, default=1000)
    parser.add_argument("--train-interval", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--train-batches", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--buffer-size", type=int, default=2_000_000)
    _default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'deep_cfr')
    parser.add_argument("--output", type=str, default=_default_out)
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    trainer = DeepCFR(lr=args.lr, buffer_size=args.buffer_size)
    trainer.run(
        num_iterations=args.iterations,
        traversals_per_iter=args.traversals,
        train_interval=args.train_interval,
        batch_size=args.batch_size,
        num_batches=args.train_batches,
    )
    trainer.export(args.output)


if __name__ == "__main__":
    main()
