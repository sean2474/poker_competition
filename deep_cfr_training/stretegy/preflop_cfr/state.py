from typing import List
from game.game import (
    BIG_BLIND, SMALL_BLIND, MAX_BET, _RAISE_SIZES,
)

class _State:
    __slots__ = ('bets', 'hist', 'acting', 'n_raises', 'done', 'folded_by')

    def __init__(self, bets=(SMALL_BLIND, BIG_BLIND), hist='',
                 acting=0, n_raises=0, done=False, folded_by=-1):
        self.bets      = tuple(bets)
        self.hist      = hist
        self.acting    = acting
        self.n_raises  = n_raises
        self.done      = done
        self.folded_by = folded_by

    def valid(self) -> List[str]:
        to_call = max(self.bets) - self.bets[self.acting]
        acts = ['f', 'c'] if to_call > 0 else ['k']
        if self.n_raises < len(_RAISE_SIZES) and max(self.bets) < MAX_BET:
            acts.append('r')
        return acts

    def apply(self, a: str) -> '_State':
        cp, op = self.acting, 1 - self.acting
        h = self.hist + a
        if a == 'f':
            return _State(self.bets, h, cp, self.n_raises, done=True, folded_by=cp)
        if a == 'k':
            return _State(self.bets, h, op, self.n_raises, done=(cp == 1))
        if a == 'c':
            nb = list(self.bets); nb[cp] = self.bets[op]
            bb_option = (cp == 0 and self.n_raises == 0)
            return _State(tuple(nb), h, op, self.n_raises, done=not bb_option)
        if a == 'r':
            nb = list(self.bets)
            nb[cp] = min(_RAISE_SIZES[self.n_raises], MAX_BET)
            return _State(tuple(nb), h, op, self.n_raises + 1)
        raise ValueError(a)