"""
ELO-style evaluation: run multiple matches against all baseline agents,
track win/loss (not chip margin), compute win rate and estimated ELO.

Usage:
    python elo_test.py                    # default: 3 matches per opponent
    python elo_test.py --matches 5        # 5 matches per opponent
"""

import argparse
import json
import logging
import multiprocessing
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from match import run_api_match


OPPONENTS = {
    "CallingStation": "agents.test_agents.CallingStationAgent",
    "AllIn": "agents.test_agents.AllInAgent",
    "Random": "agents.test_agents.RandomAgent",
    "ProbAgent": "agents.prob_agent.ProbabilityAgent",
}

BOT0_PATH = "submission.player.PlayerAgent"
PORT_BOT0 = 8000
PORT_BOT1 = 8001


def start_agent(module_path, port, player_id):
    """Start an agent server in a subprocess."""
    parts = module_path.rsplit(".", 1)
    module_name, class_name = parts[0], parts[1]
    mod = __import__(module_name, fromlist=[class_name])
    cls = getattr(mod, class_name)
    cls.run(stream=False, port=port, player_id=player_id)


def run_match_pair(opp_name, opp_path, match_idx):
    """Run one match, return (opp_name, match_idx, bankroll, win/loss/tie, time_used)."""
    logger = logging.getLogger(f"elo_{opp_name}_{match_idx}")
    logger.setLevel(logging.WARNING)

    # Start both agents
    p0 = multiprocessing.Process(target=start_agent, args=(BOT0_PATH, PORT_BOT0, "elo0"))
    p1 = multiprocessing.Process(target=start_agent, args=(opp_path, PORT_BOT1, "elo1"))
    p0.start()
    p1.start()
    time.sleep(2)

    try:
        result = run_api_match(
            f"http://localhost:{PORT_BOT0}",
            f"http://localhost:{PORT_BOT1}",
            logger,
            num_hands=1000,
            csv_path="/dev/null",
            team_0_name="PlayerAgent",
            team_1_name=opp_name,
        )
        bankroll = result.get("bot0_reward", 0)
        t_used = result.get("bot0_time_used", 0)

        if bankroll > 0:
            outcome = "WIN"
        elif bankroll < 0:
            outcome = "LOSS"
        else:
            outcome = "TIE"

        return opp_name, match_idx, bankroll, outcome, t_used

    except Exception as e:
        return opp_name, match_idx, 0, f"ERROR: {e}", 0
    finally:
        p0.terminate()
        p1.terminate()
        p0.join(timeout=3)
        p1.join(timeout=3)


def compute_elo_estimate(win_rate):
    """Estimate ELO difference from win rate. Win rate 0.5 = same ELO."""
    if win_rate <= 0:
        return -400
    if win_rate >= 1:
        return 400
    import math
    return 400 * math.log10(win_rate / (1 - win_rate))


def main():
    parser = argparse.ArgumentParser(description="ELO-style evaluation")
    parser.add_argument("--matches", type=int, default=3, help="Matches per opponent")
    parser.add_argument("--opponents", nargs="*", default=None, help="Specific opponents to test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    opponents = OPPONENTS
    if args.opponents:
        opponents = {k: v for k, v in OPPONENTS.items() if k in args.opponents}

    print(f"{'='*70}")
    print(f"  ELO EVALUATION: {args.matches} matches per opponent")
    print(f"{'='*70}")
    print()

    all_results = []
    total_wins = 0
    total_matches = 0

    for opp_name, opp_path in opponents.items():
        wins = 0
        losses = 0
        ties = 0
        bankrolls = []

        print(f"--- vs {opp_name} ---")

        for m in range(args.matches):
            opp, idx, bankroll, outcome, t_used = run_match_pair(opp_name, opp_path, m)
            bankrolls.append(bankroll)

            marker = "✅" if outcome == "WIN" else ("❌" if outcome == "LOSS" else "➖")
            print(f"  Match {m+1}: {marker} {outcome} {bankroll:+d} chips ({t_used:.1f}s)")

            if outcome == "WIN":
                wins += 1
            elif outcome == "LOSS":
                losses += 1
            else:
                ties += 1

            all_results.append((opp_name, outcome, bankroll))

        win_rate = wins / args.matches if args.matches > 0 else 0
        avg_bankroll = sum(bankrolls) / len(bankrolls) if bankrolls else 0
        elo_diff = compute_elo_estimate(win_rate) if wins + losses > 0 else 0

        print(f"  Summary: {wins}W {losses}L {ties}T  WR={win_rate*100:.0f}%  "
              f"Avg={avg_bankroll:+.0f}  ELO≈{elo_diff:+.0f}")
        print()

        total_wins += wins
        total_matches += args.matches

    # Overall
    overall_wr = total_wins / total_matches if total_matches > 0 else 0
    overall_elo = compute_elo_estimate(overall_wr) if total_matches > 0 else 0

    print(f"{'='*70}")
    print(f"  OVERALL: {total_wins}W / {total_matches} matches = {overall_wr*100:.0f}% win rate")
    print(f"  Estimated ELO advantage: {overall_elo:+.0f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
