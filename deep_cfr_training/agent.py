import numpy as np

from interface.agent import AgentBase
from interface.model import PreflopModel, DiscardModel, PostflopModel
from range_finder.core import Range


class Agent(AgentBase):
    def __init__(self, preflop: PreflopModel, discard: DiscardModel,
                 postflop: PostflopModel = None):
        super().__init__(preflop, discard, postflop)
        self.range = Range()

    # ── Hand lifecycle ────────────────────────────────────────────────────────

    def reset(self):
        self.range.reset()

    # ── Preflop ───────────────────────────────────────────────────────────────

    def act_preflop(self, hand: list, history: str) -> tuple[str, dict]:
        action, probs = self.preflop.action(hand, history)
        self.range.update(
            board=[], hero_discard=[], opp_discard=[],
            phase='preflop', model=self.preflop,
            hero_action=action, history=history,
        )
        return action, probs

    def observe_opp_preflop(self, opp_action: str, history: str):
        self.range.update(
            board=[], hero_discard=[], opp_discard=[],
            phase='preflop', model=self.preflop,
            action=opp_action, history=history,
        )

    # ── Discard ───────────────────────────────────────────────────────────────

    def act_discard(self, hand: list, board: list,
                    opp_discard: list = None) -> tuple[tuple, np.ndarray]:
        keep_idx, probs = self.discard.action(
            board, hand, '',
            self.range.hero_range, self.range.opp_range,
            opp_discard_card=opp_discard,
        )
        hero_discarded = [c for i, c in enumerate(hand) if i not in keep_idx]
        self.range.update(
            board=board, hero_discard=hero_discarded,
            opp_discard=opp_discard or [],
            phase='discard', model=self.discard,
        )
        return keep_idx, probs

    def observe_opp_discard(self, opp_discard: list, board: list):
        self.range.update(
            board=board, hero_discard=[], opp_discard=opp_discard,
            phase='discard', model=self.discard,
        )

    # ── Postflop ──────────────────────────────────────────────────────────────

    def act_postflop(self, hand: list, board: list, history: str) -> tuple[str, dict]:
        if self.postflop is None:
            return 'c', {'c': 1.0}
        action, probs = self.postflop.action(
            hand, board, history,
            self.range.hero_range, self.range.opp_range,
        )
        self.range.update(
            board=board, hero_discard=[], opp_discard=[],
            phase='postflop', model=self.postflop,
            hero_action=action, history=history,
        )
        return action, probs

    def observe_opp_postflop(self, opp_action: str, history: str, board: list):
        self.range.update(
            board=board, hero_discard=[], opp_discard=[],
            phase='postflop', model=self.postflop,
            action=opp_action, history=history,
        )

    # ── Training (range-connected) ────────────────────────────────────────────

    def train(self, phase: str = 'all',
              preflop_kwargs: dict = None,
              discard_kwargs: dict = None,
              postflop_kwargs: dict = None):
        """
        Train models in sequence, automatically wiring range dependencies:
          preflop  → standalone
          discard  → receives self.preflop as preflop_model
          postflop → receives self.preflop + self.discard
        """
        if phase in ('preflop', 'all'):
            self.preflop.train(**(preflop_kwargs or {}))

        if phase in ('discard', 'all'):
            kw = {'preflop_model': self.preflop}
            kw.update(discard_kwargs or {})
            self.discard.train(**kw)

        if phase in ('postflop', 'all') and self.postflop is not None:
            kw = {'preflop_model': self.preflop, 'discard_model': self.discard}
            kw.update(postflop_kwargs or {})
            self.postflop.train(**kw)