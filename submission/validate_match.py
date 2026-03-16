"""
Quick validation: run 200 hands vs ProbabilityAgent and report result.
Called by C++ trainer during periodic validation.
"""
import os
import sys
import logging
import multiprocessing
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from match import run_api_match

def load_and_run(agent_class, stream, port, player_id):
    agent_class.run(stream, port, player_id=player_id)

def main():
    from submission.player import PlayerAgent
    from agents.prob_agent import ProbabilityAgent

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger("validate")
    logger.setLevel(logging.INFO)

    port0, port1 = 9000, 9001

    p0 = multiprocessing.Process(target=PlayerAgent.run, args=(False, port0), kwargs={"player_id": "val0"})
    p1 = multiprocessing.Process(target=ProbabilityAgent.run, args=(False, port1), kwargs={"player_id": "val1"})

    p0.start()
    p1.start()
    time.sleep(2)  # wait for servers

    try:
        result = run_api_match(
            f"http://localhost:{port0}",
            f"http://localhost:{port1}",
            logger,
            num_hands=200,
            csv_path="/dev/null",
            team_0_name="PlayerAgent",
            team_1_name="ProbAgent",
        )
        reward = result.get("bot0_reward", 0)
        t0 = result.get("bot0_time_used", 0)
        print(f"VALIDATION: {reward:+d} chips vs ProbAgent (200 hands, {t0:.1f}s used)")
    except Exception as e:
        print(f"VALIDATION ERROR: {e}")
    finally:
        p0.terminate()
        p1.terminate()
        p0.join(timeout=3)
        p1.join(timeout=3)

if __name__ == "__main__":
    main()
