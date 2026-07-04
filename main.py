"""
DecisionCourt — adversarial multi-agent decision review.

Usage:
    python main.py "Should I do a Masters abroad or take my placement offer?"
    python main.py "Buy a car or use Uber daily?" --priorities "low cost, flexibility"
    python main.py --demo

Requires GROQ_API_KEY in a .env file (see .env.example).
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from decisioncourt.graph import build_graph, initial_state
from decisioncourt.llm import get_llm

HERE = os.path.dirname(os.path.abspath(__file__))
VERDICTS_DIR = os.path.join(HERE, "verdicts")

DEMO = [
    ("Should a final-year student take a decent placement offer now or "
     "spend six more months hunting for a better AI/ML role?",
     "career growth, financial stability, learning curve"),
    ("Should a small D2C brand invest its marketing budget in Instagram "
     "influencers or in Google performance ads?",
     "measurable ROI, brand building on a small budget"),
]


def run(decision, priorities=""):
    llm = get_llm()
    graph = build_graph(llm, VERDICTS_DIR)
    final = graph.invoke(initial_state(decision, priorities),
                         config={"recursion_limit": 50})
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    v = final["verdict"]
    print(f"Recommendation: {v['recommendation']}")
    print(f"Confidence:     {v['confidence']}%")
    for i, r in enumerate(v["reasoning"], 1):
        print(f"  {i}. {r}")
    print("Flips if:")
    for f in v["flips_if"]:
        print(f"  - {f}")
    print(f"\nFull memo: {final['memo_path']}")
    return final


def main():
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not set. Copy .env.example to .env and add "
                 "your free key from https://console.groq.com")

    ap = argparse.ArgumentParser(description="DecisionCourt")
    ap.add_argument("decision", nargs="?", help="the decision to review")
    ap.add_argument("--priorities", default="",
                    help="what you care about, comma-separated")
    ap.add_argument("--demo", action="store_true",
                    help="run 2 showcase decisions")
    args = ap.parse_args()

    if args.demo:
        for d, p in DEMO:
            run(d, p)
    elif args.decision:
        run(args.decision, args.priorities)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
