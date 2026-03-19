"""
Preflop fallback frequency check — direct lookup test (no subprocess).

Simulates every combination of (size_bucket, history, 1000 random hands)
and reports how often each path is taken.

Usage: python check_preflop_fallback.py
"""
import sys, os, pickle, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'submission'))

from features import NUM_RANKS
from action import BIG_BLIND, FOLD, RAISE, CALL, CHECK
from strategy.preflop import (
    canonicalize, preflop_key, size_bucket,
    get_fallback_stats, reset_fallback_stats, preflop_action,
)

CHART_PATH = os.path.join(os.path.dirname(__file__),
                          'submission', 'model', 'deep_cfr_preflop_chart.pkl')
with open(CHART_PATH, 'rb') as f:
    chart = pickle.load(f)
print(f"Chart loaded: {len(chart):,} keys")

# All 27 cards, all 5-card hands (sampled)
all_cards = list(range(NUM_RANKS * 3))

# Bet sizes to test: representative of what opponents might bet
# 's' range: ≤2.6bb=5.2 chips,  'L': >4.1bb=8.2 chips
TEST_BETS = {
    'normal_open_2.5bb': 5,      # training size → 's' bucket
    'small_open_1bb':    2,      # limp → 's'
    'large_open_5bb':    10,     # off-size → 'L'
    'allin_first':       1000,   # AllIn → 'L'
    '3bet_training':     15,     # normal 3-bet → 'L'
    '3bet_tiny':         6,      # tiny 3-bet → 's'
}

# History strings that appear at each bucket
HIST_BY_BUCKET = {
    's': ['', 'r', 'c', 'cb'],
    'L': ['rr', 'cbr', 'rrr', 'cbrr'],
}

def make_obs(bet, valid_raise=True):
    return {
        'my_bet':      0,
        'opp_bet':     bet,
        'my_cards':    [0, 1, 2, 3, 4],  # placeholder; hand5 injected per call
        'valid_actions': {FOLD: True, CALL: True, CHECK: False, RAISE: valid_raise},
        'min_raise':   bet * 2,
        'max_raise':   1000,
    }

# ── Per-scenario coverage table ──────────────────────────────────────────────
print("\n=== Coverage table (100 random hands per cell) ===")
print(f"{'scenario':<22} {'hist':<6} {'bucket':<6} {'hit':>5} {'adj':>5} {'miss':>5}")
print("-" * 55)

N_HANDS = 100
for label, bet in TEST_BETS.items():
    bkt = size_bucket(bet)
    hists = HIST_BY_BUCKET.get(bkt, [''])
    for hist in hists:
        reset_fallback_stats()
        for _ in range(N_HANDS):
            hand5 = random.sample(all_cards, 5)
            obs = make_obs(bet)
            obs['my_cards'] = hand5
            # Fake action_history that produces the desired history string
            # We bypass action_history encoding and patch key directly
            key = (canonicalize(hand5), bkt, hist)
            strat = chart.get(key)
            from strategy.preflop import _stats, _adjust_strat_for_size, _TRAIN_BET
            _stats['total'] += 1
            if strat is None or strat.sum() <= 0:
                other = 'L' if bkt == 's' else 's'
                base  = chart.get((key[0], other, hist))
                if base is not None and base.sum() > 0:
                    _stats['size_adj'] += 1
                else:
                    _stats['blind_call'] += 1
            else:
                _stats['hit'] += 1
        s = get_fallback_stats()
        print(f"{label:<22} {hist!r:<6} {bkt!r:<6} {s['chart_hit']:>5} {s['size_adj']:>5} {s['blind_call']:>5}")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== Summary: AllIn (worst case) hits ===")
reset_fallback_stats()
for hist_list in HIST_BY_BUCKET.values():
    for hist in hist_list:
        for _ in range(N_HANDS):
            hand5 = random.sample(all_cards, 5)
            key   = (canonicalize(hand5), 'L', hist)   # AllIn → always 'L'
            strat = chart.get(key)
            from strategy.preflop import _stats
            _stats['total'] += 1
            if strat is None or strat.sum() <= 0:
                base = chart.get((key[0], 's', hist))
                if base is not None and base.sum() > 0:
                    _stats['size_adj'] += 1
                else:
                    _stats['blind_call'] += 1
            else:
                _stats['hit'] += 1

s = get_fallback_stats()
print(f"  total={s['total']}  hit={s['chart_hit']} ({100*s['chart_hit']/s['total']:.1f}%)"
      f"  size_adj={s['size_adj']} ({100*s['size_adj']/s['total']:.1f}%)"
      f"  blind_call={s['blind_call']} ({100*s['blind_call']/s['total']:.1f}%)")
