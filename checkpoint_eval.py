"""
Evaluate multiple checkpoints by win rate.

Usage:
    # Evaluate a single checkpoint binary
    python checkpoint_eval.py --bin training/data/strategy_cpp.bin --matches 3

    # Evaluate all .bin files in a directory (e.g. saved at different iterations)
    python checkpoint_eval.py --dir training/data/checkpoints/ --matches 2

    # Compare current strategy vs no strategy (fallback only)
    python checkpoint_eval.py --compare-fallback --matches 3
"""

import argparse
import glob
import logging
import multiprocessing
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from match import run_api_match

OPPONENTS = {
    "ProbAgent": "agents.prob_agent.ProbabilityAgent",
}

BOT0_PATH = "submission.player.PlayerAgent"
PORT0 = 8000
PORT1 = 8001


def convert_bin_to_numpy(bin_path, out_dir):
    """Convert C++ binary to numpy strategy files."""
    import pickle
    import numpy as np

    ACTION_LISTS = [
        ("FOLD", "CALL", "JAM"),
        ("FOLD", "CALL"),
        ("FOLD", "CALL", "RAISE_SMALL", "RAISE_LARGE"),
        ("CHECK", "BET_SMALL", "BET_LARGE"),
        ("CHECK",),
    ]
    MAX_ACTIONS = 4

    with open(bin_path, 'rb') as f:
        data = f.read()

    offset = 0
    iters, num_nodes = struct.unpack_from('<II', data, offset)
    offset += 8

    # Detect format by per-node byte size
    per_node = (len(data) - 8) / num_nodes if num_nodes > 0 else 42
    has_confidence = (per_node >= 49.5)  # ~50 bytes = new format with confidence

    keys = np.zeros(num_nodes, dtype=np.uint64)
    act_types = np.zeros(num_nodes, dtype=np.uint8)
    probs = np.zeros((num_nodes, MAX_ACTIONS), dtype=np.uint8)

    for i in range(num_nodes):
        key = struct.unpack_from('<Q', data, offset)[0]; offset += 8
        atype = struct.unpack_from('<B', data, offset)[0]; offset += 1
        nact = struct.unpack_from('<B', data, offset)[0]; offset += 1
        avg = struct.unpack_from(f'<{MAX_ACTIONS}d', data, offset); offset += MAX_ACTIONS * 8
        if has_confidence:
            offset += 8  # skip confidence

        keys[i] = key
        act_types[i] = atype
        nact = min(nact, MAX_ACTIONS)
        if nact == 0:
            continue
        raw = list(avg[:nact])
        total = sum(raw)
        if total > 0:
            raw = [p / total for p in raw]
        else:
            raw = [1.0 / nact] * nact
        for j in range(nact):
            probs[i, j] = max(0, min(255, int(round(raw[j] * 255))))
        row_sum = sum(probs[i, :nact])
        if row_sum > 0 and row_sum != 255 and nact > 0:
            mx = np.argmax(probs[i, :nact])
            probs[i, mx] = max(0, min(255, probs[i, mx] + (255 - row_sum)))

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "strategy_keys.npy"), keys)
    np.save(os.path.join(out_dir, "strategy_acttype.npy"), act_types)
    np.save(os.path.join(out_dir, "strategy_probs.npy"), probs)

    meta = {'iterations': iters, 'num_nodes': num_nodes,
            'action_lists': ACTION_LISTS, 'max_actions': MAX_ACTIONS}
    import pickle
    with open(os.path.join(out_dir, "strategy_meta.pkl"), 'wb') as f:
        pickle.dump(meta, f)

    return iters, num_nodes


def start_agent(module_path, port, player_id):
    parts = module_path.rsplit(".", 1)
    mod = __import__(parts[0], fromlist=[parts[1]])
    cls = getattr(mod, parts[1])
    cls.run(stream=False, port=port, player_id=player_id)


def run_one_match(opp_name, opp_path):
    logger = logging.getLogger("ckpt_eval")
    logger.setLevel(logging.WARNING)

    p0 = multiprocessing.Process(target=start_agent, args=(BOT0_PATH, PORT0, "e0"))
    p1 = multiprocessing.Process(target=start_agent, args=(opp_path, PORT1, "e1"))
    p0.start(); p1.start()
    time.sleep(2)

    try:
        result = run_api_match(
            f"http://localhost:{PORT0}", f"http://localhost:{PORT1}",
            logger, num_hands=1000, csv_path="/dev/null",
            team_0_name="Player", team_1_name=opp_name,
        )
        bankroll = result.get("bot0_reward", 0)
        return "WIN" if bankroll > 0 else ("LOSS" if bankroll < 0 else "TIE"), bankroll
    except Exception as e:
        print(f"    MATCH ERROR: {e}")
        return "LOSS", 0
    finally:
        p0.terminate(); p1.terminate()
        p0.join(timeout=3); p1.join(timeout=3)
        import subprocess
        subprocess.run(["pkill", "-f", f"uvicorn.*{PORT0}"], capture_output=True)
        subprocess.run(["pkill", "-f", f"uvicorn.*{PORT1}"], capture_output=True)
        time.sleep(1)


def eval_checkpoint(bin_path, num_matches, data_dir):
    """Evaluate a single checkpoint. Returns (iters, nodes, wins, losses, avg_bankroll)."""
    # Convert binary to numpy in submission/data
    iters, nodes = convert_bin_to_numpy(bin_path, data_dir)
    print(f"\n  Checkpoint: {iters:,} iters, {nodes:,} nodes")

    wins = losses = 0
    bankrolls = []

    for opp_name, opp_path in OPPONENTS.items():
        for m in range(num_matches):
            outcome, bankroll = run_one_match(opp_name, opp_path)
            bankrolls.append(bankroll)
            marker = "✅" if outcome == "WIN" else ("❌" if outcome == "LOSS" else "➖")
            print(f"    vs {opp_name} #{m+1}: {marker} {outcome} {bankroll:+d}")
            if outcome == "WIN": wins += 1
            elif outcome == "LOSS": losses += 1

    total = wins + losses
    wr = wins / total * 100 if total > 0 else 0
    avg = sum(bankrolls) / len(bankrolls) if bankrolls else 0
    print(f"  Result: {wins}W/{losses}L  WR={wr:.0f}%  Avg={avg:+.0f}")

    return iters, nodes, wins, losses, avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", type=str, help="Single binary file to evaluate")
    parser.add_argument("--dir", type=str, help="Directory of checkpoint .bin files")
    parser.add_argument("--matches", type=int, default=2, help="Matches per opponent per checkpoint")
    parser.add_argument("--compare-fallback", action="store_true", help="Also test with no strategy (fallback only)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    data_dir = os.path.join(os.path.dirname(__file__), "submission", "data")

    results = []

    if args.compare_fallback:
        print("="*60)
        print("  FALLBACK ONLY (no CFR strategy)")
        print("="*60)
        # Remove strategy files
        for f in ["strategy_keys.npy", "strategy_acttype.npy", "strategy_probs.npy", "strategy_meta.pkl"]:
            p = os.path.join(data_dir, f)
            if os.path.exists(p):
                os.rename(p, p + ".bak")
        try:
            wins = losses = 0
            bankrolls = []
            for opp_name, opp_path in OPPONENTS.items():
                for m in range(args.matches):
                    outcome, bankroll = run_one_match(opp_name, opp_path)
                    bankrolls.append(bankroll)
                    marker = "✅" if outcome == "WIN" else "❌"
                    print(f"  vs {opp_name} #{m+1}: {marker} {outcome} {bankroll:+d}")
                    if outcome == "WIN": wins += 1
                    elif outcome == "LOSS": losses += 1
            total = wins + losses
            wr = wins / total * 100 if total > 0 else 0
            print(f"  Result: {wins}W/{losses}L  WR={wr:.0f}%")
            results.append(("fallback", 0, 0, wins, losses, sum(bankrolls)/len(bankrolls) if bankrolls else 0))
        finally:
            for f in ["strategy_keys.npy", "strategy_acttype.npy", "strategy_probs.npy", "strategy_meta.pkl"]:
                p = os.path.join(data_dir, f)
                if os.path.exists(p + ".bak"):
                    os.rename(p + ".bak", p)

    if args.bin:
        print("="*60)
        print(f"  EVALUATING: {args.bin}")
        print("="*60)
        r = eval_checkpoint(args.bin, args.matches, data_dir)
        results.append(("checkpoint", *r))

    if args.dir:
        bins = sorted(glob.glob(os.path.join(args.dir, "*.bin")))
        if not bins:
            print(f"No .bin files found in {args.dir}")
            return
        print("="*60)
        print(f"  EVALUATING {len(bins)} CHECKPOINTS from {args.dir}")
        print("="*60)
        for bp in bins:
            r = eval_checkpoint(bp, args.matches, data_dir)
            results.append((os.path.basename(bp), *r))

    # Summary table
    if len(results) > 1:
        print("\n" + "="*60)
        print(f"  {'Name':<25} {'Iters':>10} {'Nodes':>8} {'W':>3} {'L':>3} {'WR':>5} {'Avg':>8}")
        print("-"*60)
        for name, iters, nodes, w, l, avg in results:
            total = w + l
            wr = w / total * 100 if total > 0 else 0
            print(f"  {name:<25} {iters:>10,} {nodes:>8,} {w:>3} {l:>3} {wr:>4.0f}% {avg:>+8.0f}")
        print("="*60)


if __name__ == "__main__":
    main()
