"""
test_discard_vs_ev.py
DiscardNet 선택 vs 단순 MC equity 비교.

각 핸드에 대해:
  - DiscardNet strategy (확률분포) + 샘플 선택
  - MC equity (10가지 keep pair 각각 400 sims)
  - 최대 EV pair와 DiscardNet 선택 비교

Usage:
    cd /Users/sean2474/Desktop/project/poker-engine-2026
    python test_discard_vs_ev.py --hands 200
"""

import sys, os, argparse, random
sys.path.insert(0, '.')
sys.path.insert(0, 'submission')

import numpy as np
import torch

from gym_env import PokerEnv
from submission.strategy.discard import decide_discard, build_discard_feats, _opp_cats_from_obs
from submission.action import DiscardNet
from features import KEEP_PAIRS

DECK_SIZE  = 27
N_SIMS     = 400


# ── MC equity ─────────────────────────────────────────────────────────────────

def mc_equity(keep2, community, opp_disc, evaluator, n_sims=N_SIMS):
    shown = set(keep2) | set(c for c in community if c >= 0) | set(c for c in opp_disc if c >= 0)
    pool  = [c for c in range(DECK_SIZE) if c not in shown]
    board_known = [c for c in community if c >= 0]
    board_needed = 5 - len(board_known)
    wins = valid = 0
    for _ in range(n_sims):
        if len(pool) < 2 + board_needed:
            break
        sample = random.sample(pool, 2 + board_needed)
        opp2   = sample[:2]
        full_board = board_known + sample[2:]
        if len(full_board) != 5:
            continue
        my_h  = list(map(PokerEnv.int_to_card, keep2))
        op_h  = list(map(PokerEnv.int_to_card, opp2))
        board = list(map(PokerEnv.int_to_card, full_board))
        my_r  = evaluator.evaluate(my_h, board)
        op_r  = evaluator.evaluate(op_h, board)
        if my_r < op_r: wins += 1
        valid += 1
    return wins / valid if valid > 0 else 0.5


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hands', type=int, default=200)
    parser.add_argument('--model', default='submission/model/deep_cfr_full.pt')
    args = parser.parse_args()

    env = PokerEnv()
    evaluator = env.evaluator

    ckpt = torch.load(args.model, map_location='cpu')
    h    = ckpt.get('discard_hidden', 128)
    net  = DiscardNet(h)
    net.load_state_dict(ckpt['discard_net'])
    net.eval()
    print(f'DiscardNet loaded (hidden={h}, iter={ckpt.get("discard_iteration","?")})')

    ev_diffs   = []     # EV(net_choice) - EV(best_choice)
    agree_best = 0      # net sampled == best EV pair
    agree_top3 = 0      # net sampled in top-3 EV pairs

    rng = np.random.default_rng(0)

    for hand_i in range(args.hands):
        # Deal a hand (5 cards + 3 flop)
        deck = list(range(DECK_SIZE))
        random.shuffle(deck)
        hand5  = deck[:5]
        board3 = deck[5:8]
        opp5   = deck[8:13]
        opp_disc_idxs = random.sample(range(5), 3)
        opp_disc = [opp5[i] for i in opp_disc_idxs]

        obs = {
            'my_cards':          hand5 + [-1]*0,
            'community_cards':   board3 + [-1]*2,
            'opp_discarded_cards': opp_disc,
            'acting_agent':      random.choice([0, 1]),
        }

        # ── MC equity for all 10 pairs ──────────────────────────────────────
        equities = []
        for ai, aj in KEEP_PAIRS:
            keep2 = [hand5[ai], hand5[aj]]
            eq    = mc_equity(keep2, board3, opp_disc, evaluator, N_SIMS)
            equities.append(eq)
        equities = np.array(equities)
        best_k   = int(np.argmax(equities))
        sorted_k = np.argsort(equities)[::-1]

        # ── DiscardNet strategy ─────────────────────────────────────────────
        opp_cats = _opp_cats_from_obs(hand5, board3, obs)
        is_bb    = obs['acting_agent'] == 1
        feats    = build_discard_feats(hand5, board3, opp_cats, is_bb)
        strat    = net.get_strategy(feats.astype(np.float32))   # (10,)
        strat    = strat.astype(np.float64); strat /= strat.sum()

        # Sample DiscardNet's choice
        net_k = int(rng.choice(10, p=strat))

        # EV of net's choice vs best
        ev_net  = equities[net_k]
        ev_best = equities[best_k]
        ev_diffs.append(ev_net - ev_best)

        if net_k == best_k:                      agree_best += 1
        if net_k in sorted_k[:3]:                agree_top3 += 1

        if (hand_i + 1) % 50 == 0:
            print(f'  [{hand_i+1}/{args.hands}] '
                  f'agree_best={agree_best/(hand_i+1):.1%}  '
                  f'agree_top3={agree_top3/(hand_i+1):.1%}  '
                  f'avg_ev_diff={np.mean(ev_diffs):.4f}')

    ev_diffs = np.array(ev_diffs)
    print(f'\n{"="*55}')
    print(f' DiscardNet vs Pure EV  ({args.hands} hands, {N_SIMS} sims/pair)')
    print(f'{"="*55}')
    print(f'  Best EV 선택 일치율:   {agree_best/args.hands:.1%}')
    print(f'  Top-3 EV 내 선택율:    {agree_top3/args.hands:.1%}')
    print(f'  평균 EV 차이 (net-best): {ev_diffs.mean():.4f}')
    print(f'  EV 차이 std:            {ev_diffs.std():.4f}')
    print(f'  EV 차이 p5~p95:         [{np.percentile(ev_diffs,5):.4f}, {np.percentile(ev_diffs,95):.4f}]')
    print()
    if ev_diffs.mean() > -0.02:
        print('  → 정상 범위 (EV 손실 < 2%)')
    elif ev_diffs.mean() > -0.05:
        print('  → 약간 벗어남 (GTO 전략 학습 중)')
    else:
        print('  → 많이 벗어남 — 학습 문제 가능성 있음')


if __name__ == '__main__':
    main()
