"""
1000-hand matches vs FoldAgent, CallingStationAgent, AllInAgent, RandomAgent, ProbabilityAgent
Usage: python run_match_1000.py [--hands 1000]
"""
import argparse, logging, multiprocessing, sys, time

def main():
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument('--hands', type=int, default=1000)
    args = parser.parse_args()

    from agents.test_agents import FoldAgent, CallingStationAgent, AllInAgent, RandomAgent
    from agents.prob_agent   import ProbabilityAgent
    from submission.player   import PlayerAgent
    import match as match_module
    from match import run_api_match

    logger = logging.getLogger('match')

    OPPONENTS = [
        ('FoldAgent',           FoldAgent),
        ('CallingStationAgent', CallingStationAgent),
        ('AllInAgent',          AllInAgent),
        ('RandomAgent',         RandomAgent),
        ('ProbabilityAgent',    ProbabilityAgent),
    ]

    print(f"Running {args.hands} hands vs each opponent\n")
    print(f"{'Opponent':<22} {'PlayerAgent':>12} {'Opponent':>12} {'Result':>7} {'sec/hand':>9}")
    print("-" * 68)

    for name, OppClass in OPPONENTS:
        match_module.bankrolls = [0, 0]
        match_module.time_used_0 = 0.0
        match_module.time_used_1 = 0.0
        match_module.failure_tracker.failed_attempts = {0: 0, 1: 0}

        p0 = multiprocessing.Process(target=PlayerAgent.run,  args=(False, 8000))
        p1 = multiprocessing.Process(target=OppClass.run,     args=(False, 8001))
        p0.start(); p1.start()
        time.sleep(2)
        t0 = time.time()
        try:
            result = run_api_match("http://127.0.0.1:8000", "http://127.0.0.1:8001",
                                   logger, num_hands=args.hands,
                                   csv_path=f"match_{name}_{args.hands}.csv")
        except Exception as e:
            print(f"{name:<22}  ERROR: {e}")
            p0.terminate(); p1.terminate(); p0.join(); p1.join()
            continue
        finally:
            p0.terminate(); p1.terminate()
            p0.join(); p1.join()

        elapsed = time.time() - t0
        p0c = result.get('bot0_reward', 0)
        p1c = result.get('bot1_reward', 0)
        res = result.get('result', result.get('status', '?'))
        sph = elapsed / args.hands
        sign = lambda v: f"+{v}" if v > 0 else str(v)
        print(f"{name:<22} {sign(p0c):>12} {sign(p1c):>12} {res:>7} {sph:>9.3f}s")


if __name__ == '__main__':
    main()
