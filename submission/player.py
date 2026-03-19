"""
player.py — Deep CFR PlayerAgent.

Strategy routing:
  DISCARD  → strategy/discard.py  (DiscardNet or fast_score heuristic)
  street 0 → strategy/preflop.py  (tabular CFR chart)
  street 1-3 → strategy/postflop.py (StrategyNet 77-dim)

Models loaded at init from model/:
  deep_cfr_strategy.pt          postflop StrategyNet
  deep_cfr_preflop_chart.pkl    preflop tabular chart
  deep_cfr_full.pt              (optional) includes discard_net
"""

import os
import sys
import pickle
import torch

# Ensure submission/ dir is on path so 'action', 'features', 'strategy.*' resolve
_SUBMISSION_DIR = os.path.dirname(os.path.abspath(__file__))
if _SUBMISSION_DIR not in sys.path:
    sys.path.insert(0, _SUBMISSION_DIR)

from agents.agent import Agent
from gym_env import PokerEnv

from action import DISCARD, FOLD, RAISE, CHECK, CALL, StrategyNet, DiscardNet
from strategy.preflop      import preflop_action, size_bucket, canonicalize
from strategy.discard      import decide_discard
from strategy.postflop     import postflop_action
from strategy.range_tracker import OppRangeTracker

_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')


class PlayerAgent(Agent):

    def __init__(self, stream: bool = True):
        super().__init__(stream)
        self.action_types = PokerEnv.ActionType
        self._strategy_net  = self._load_strategy_net()
        self._preflop_chart = self._load_preflop_chart()
        self._discard_net   = self._load_discard_net()

        # Per-hand state
        self._my_id         = -1
        self._hand_number   = -1
        self._pf_history    = []      # training action indices this hand
        self._pf_hist_str   = ''      # string form for range tracker
        self._my_disc       = [-1,-1,-1]
        self._opp_disc      = [-1,-1,-1]
        self._prev_street   = -1
        self._aggressor_me  = False
        self._aggressor_opp = False
        self._n_bets_me     = 0
        self._n_bets_opp    = 0
        self._prev_my_bet   = 0
        self._prev_opp_bet  = 0
        self._opp_range     = OppRangeTracker()
        self._pf_updates_applied = False

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_strategy_net(self):
        p = os.path.join(_MODEL_DIR, 'deep_cfr_strategy.pt')
        assert os.path.exists(p), f'strategy net not found: {p}'
        net = StrategyNet()
        net.load_state_dict(torch.load(p, map_location='cpu'))
        net.eval()
        self.logger.info('strategy net loaded')
        return net

    def _load_preflop_chart(self):
        p = os.path.join(_MODEL_DIR, 'deep_cfr_preflop_chart.pkl')
        assert os.path.exists(p), f'preflop chart not found: {p}'
        with open(p, 'rb') as f:
            chart = pickle.load(f)
        assert len(chart) > 0, 'preflop chart is empty'
        self.logger.info(f'preflop chart loaded: {len(chart)} infosets')
        return chart

    def _load_discard_net(self):
        p = os.path.join(_MODEL_DIR, 'deep_cfr_full.pt')
        assert os.path.exists(p), f'discard net checkpoint not found: {p}'
        ckpt = torch.load(p, map_location='cpu')
        assert 'discard_net' in ckpt, f"'discard_net' key missing from checkpoint: {p}"
        h = ckpt.get('discard_hidden', 128)
        net = DiscardNet(h)
        net.load_state_dict(ckpt['discard_net'])
        net.eval()
        self.logger.info('discard net loaded')
        return net

    # ── Hand state tracking ───────────────────────────────────────────────────

    def _reset_hand(self):
        self._pf_history   = []
        self._pf_hist_str  = ''
        self._my_disc      = [-1,-1,-1]
        self._opp_disc     = [-1,-1,-1]
        self._prev_street  = -1
        self._aggressor_me = False; self._aggressor_opp = False
        self._n_bets_me    = 0;     self._n_bets_opp    = 0
        self._prev_my_bet  = 0;     self._prev_opp_bet  = 0
        self._pf_updates_applied = False

    def _update_state(self, obs: dict):
        """Called at start of each act() to track per-hand context."""
        s = obs['street']
        if s != self._prev_street:
            self._n_bets_me  = 0
            self._n_bets_opp = 0
            self._aggressor_me = self._aggressor_opp = False
            self._prev_street = s

        my_bet  = obs['my_bet']
        opp_bet = obs['opp_bet']
        if my_bet > self._prev_my_bet:
            self._aggressor_me  = True
            self._aggressor_opp = False
            self._n_bets_me    += 1
        if opp_bet > self._prev_opp_bet:
            self._aggressor_opp = True
            self._aggressor_me  = False
            self._n_bets_opp   += 1
        self._prev_my_bet  = my_bet
        self._prev_opp_bet = opp_bet

        # Update known discards
        my_d  = obs.get('my_discarded_cards',  [-1,-1,-1])
        opp_d = obs.get('opp_discarded_cards', [-1,-1,-1])
        if any(c >= 0 for c in my_d):  self._my_disc  = list(my_d)
        if any(c >= 0 for c in opp_d):
            prev_opp_disc = self._opp_disc
            self._opp_disc = list(opp_d)
            # First time we see opp discards: update tracker
            if not any(c >= 0 for c in prev_opp_disc):
                self._opp_range.update_discard(self._opp_disc)

    # ── Main entry point ──────────────────────────────────────────────────────

    def __name__(self):
        return 'PlayerAgent'

    def act(self, observation, reward, terminated, truncated, info):
        obs = observation

        # Detect new hand
        hand_num = info.get('hand_number', self._hand_number)
        if hand_num != self._hand_number:
            self._hand_number = hand_num
            self._reset_hand()
            # Init range tracker with our 5 preflop cards
            hand5 = [c for c in obs.get('my_cards', []) if c >= 0]
            if len(hand5) == 5:
                self._opp_range.reset(hand5)

        # Identify my player ID (once)
        if self._my_id < 0:
            self._my_id = obs.get('acting_agent', 0)

        self._update_state(obs)
        v      = obs['valid_actions']
        street = obs.get('street', 0)

        self.logger.debug(f'h={hand_num} s={street} my_bet={obs["my_bet"]} opp_bet={obs["opp_bet"]}')

        # ── Discard ───────────────────────────────────────────────────────────
        if v[DISCARD]:
            result = decide_discard(obs, self._discard_net)
            self._my_disc = [c for c in obs['my_cards'] if c >= 0
                             and c not in (obs['my_cards'][result[2]],
                                          obs['my_cards'][result[3]])]
            return result

        # ── Preflop ───────────────────────────────────────────────────────────
        if street == 0:
            result = preflop_action(obs, self._preflop_chart, self._pf_history)
            at = result[0]
            training_a = {0: 0, 3: 1, 2: 2, 1: 3}
            self._pf_history.append(training_a.get(at, 2))
            _A_CH = {0: 'f', 1: 'c', 2: 'k', 3: 'b', 4: 'B', 5: 'r', 6: 'R', 7: 'p'}
            self._pf_hist_str = ''.join(_A_CH.get(a, '?') for a in self._pf_history)
            return result

        # ── Postflop (flop/turn/river) ────────────────────────────────────────
        is_bb = (self._my_id != self._hand_number % 2)

        # Apply preflop range updates once (first postflop call after discards known)
        if not self._pf_updates_applied and any(c >= 0 for c in self._opp_disc):
            self._opp_range.apply_preflop_updates(
                self._opp_disc, self._preflop_chart, canonicalize)
            self._pf_updates_applied = True

        opp_range_cats = self._opp_range.get_cats(
            obs.get('community_cards', [-1]*5),
            sum(1 for c in obs.get('community_cards', []) if c >= 0),
        )
        return postflop_action(
            obs, self._strategy_net,
            my_id         = self._my_id,
            my_disc       = self._my_disc,
            opp_disc      = self._opp_disc,
            aggressor_me  = self._aggressor_me,
            aggressor_opp = self._aggressor_opp,
            n_bets_me     = self._n_bets_me,
            n_bets_opp    = self._n_bets_opp,
            is_bb         = is_bb,
            opp_range_cats = opp_range_cats,
        )

    def observe(self, observation, reward, terminated, truncated, info):
        """Track opponent actions when it's not our turn."""
        if terminated:
            return
        obs     = observation
        street  = obs.get('street', 0)
        prev_mb = self._prev_my_bet    # before _update_state
        prev_ob = self._prev_opp_bet
        self._update_state(obs)

        # Detect opponent action from bet change
        opp_bet = obs.get('opp_bet', 0)
        my_bet  = obs.get('my_bet', 0)

        if street == 0:
            # Preflop: detect opp action from bet changes
            if opp_bet > prev_ob:
                max_bet = max(my_bet, opp_bet)
                bkt = size_bucket(max_bet)
                self._opp_range.record_preflop_action(
                    RAISE, bkt, self._pf_hist_str)
            # fold detection is implicit (hand ends)
        else:
            # Postflop: run Bayesian update with StrategyNet
            if not any(c >= 0 for c in self._opp_disc):
                return   # don't update until we know opp discards
            is_bb     = (self._my_id != self._hand_number % 2)
            opp_is_bb = not is_bb

            # Determine action type from bet change
            if opp_bet > prev_ob:
                at = RAISE
            elif my_bet > 0 and opp_bet == prev_ob:
                at = CALL
            else:
                at = CHECK

            board = obs.get('community_cards', [-1]*5)
            n_board = sum(1 for c in board if c >= 0)
            self._opp_range.update_postflop_action(
                action_type   = at,
                strategy_net  = self._strategy_net,
                my_bet        = my_bet,
                opp_bet       = opp_bet,
                board         = board,
                n_board       = n_board,
                street        = street,
                opp_is_bb     = opp_is_bb,
                my_disc       = self._my_disc,
                opp_disc      = self._opp_disc,
                aggressor_me  = self._aggressor_me,
                aggressor_opp = self._aggressor_opp,
                n_bets_me     = self._n_bets_me,
                n_bets_opp    = self._n_bets_opp,
            )

