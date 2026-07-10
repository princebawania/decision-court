"""
Evaluation harness for DecisionCourt (adversarial multi-agent decision review).

A debate/verdict system has no single "correct" answer, so we measure the
things that DO matter for a trustworthy decision aid:

  WELL-FORMEDNESS   every run yields recommendation + confidence(50-95)
                    + reasoning + flip conditions
  ORDER-INVARIANCE  run each dilemma with the two options in BOTH orders;
                    a sound system should pick the same option regardless of
                    which side is argued first (position-bias test)
  FACT-CHECK        avg claims/run and the share of non-"unverified" claims
   COVERAGE         that actually carry a source URL
  OFFLINE           with web search stubbed to [], it should still produce a
   ROBUSTNESS       valid verdict with all claims marked "unverified"
                    (validates the graceful-degradation design; no network)

Usage
-----
Put this file + decision_cases.json in an `eval/` folder in the repo, then:

    python eval/evaluate.py                 # all cases (each run twice for order test)
    python eval/evaluate.py --n 3           # only first 3 cases (faster)
    python eval/evaluate.py --self-consistency 3   # also run 1 case 3x for stability
    python eval/evaluate.py --skip-offline  # skip the search-stubbed test

Needs GROQ_API_KEY in .env (same as the app). Each full run is ~1-2 min.
Writes eval/dc_eval_results.json + eval/dc_eval_report.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == "eval" else HERE
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv          # noqa: E402
load_dotenv()

from decisioncourt.graph import build_graph, initial_state   # noqa: E402
from decisioncourt.llm import get_llm                          # noqa: E402

VERDICTS_DIR = str(HERE / "_eval_verdicts")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def load_cases() -> list[dict]:
    return json.loads((HERE / "decision_cases.json").read_text(encoding="utf-8"))["cases"]


def verdict_wellformed(v: dict) -> bool:
    if not isinstance(v, dict):
        return False
    rec = str(v.get("recommendation", "")).strip()
    conf = v.get("confidence", None)
    reasoning = v.get("reasoning", [])
    flips = v.get("flips_if", [])
    return (bool(rec)
            and isinstance(conf, int) and 50 <= conf <= 95
            and isinstance(reasoning, list) and len(reasoning) >= 1
            and isinstance(flips, list) and len(flips) >= 1)


def classify_choice(llm, recommendation: str, x: str, y: str) -> str:
    """Map a free-text recommendation onto option X, Y, or U(nclear)."""
    rec_low = (recommendation or "").lower()
    tossup_signals = ("too close", "toss-up", "tossup", "not decisive",
                      "closely matched", "depends on your own", "neither option",
                      "order-sensitive", "cannot decide", "can't decide")
    if any(sig in rec_low for sig in tossup_signals):
        return "T"   # honest toss-up (counterbalanced judge couldn't separate them)
    prompt = (
        "Two options were considered:\n"
        f"  X = {x}\n  Y = {y}\n\n"
        f"A judge gave this recommendation:\n\"{recommendation}\"\n\n"
        "Which option does the recommendation choose? "
        "Reply with EXACTLY one letter: X, Y, or U (if unclear). No other text."
    )
    try:
        out = llm.invoke(prompt).content.strip().upper()
    except Exception:
        return "U"
    for ch in out:
        if ch in ("X", "Y", "U"):
            return ch
    return "U"


def factcheck_stats(final: dict) -> dict:
    fcs = final.get("fact_checks", []) or []
    counts = Counter(fc.get("verdict", "unverified") for fc in fcs)
    non_unverified = [fc for fc in fcs if fc.get("verdict") in ("supported", "contradicted")]
    with_source = [fc for fc in non_unverified
                   if any(u for u in (fc.get("sources") or []))]
    return {
        "n_claims": len(fcs),
        "supported": counts.get("supported", 0),
        "contradicted": counts.get("contradicted", 0),
        "unverified": counts.get("unverified", 0),
        "n_non_unverified": len(non_unverified),
        "n_non_unverified_with_source": len(with_source),
    }


def run_case(graph, decision: str, priorities: str) -> dict | None:
    try:
        return graph.invoke(initial_state(decision, priorities),
                            config={"recursion_limit": 50})
    except Exception as e:                       # noqa: BLE001
        print(f"   [run FAILED] {e}")
        return None


def _patch_search_empty():
    """Force every web_search reference in the package to return []."""
    def _empty(*_a, **_k):
        return []
    patched = []
    for name, mod in list(sys.modules.items()):
        if name.startswith("decisioncourt") and getattr(mod, "web_search", None) is not None:
            patched.append((mod, mod.web_search))
            mod.web_search = _empty
    return patched


def _unpatch_search(patched):
    for mod, orig in patched:
        mod.web_search = orig


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit number of cases (0 = all)")
    ap.add_argument("--self-consistency", type=int, default=0,
                    help="also run the first case this many times to measure stability")
    ap.add_argument("--skip-offline", action="store_true",
                    help="skip the search-stubbed robustness test")
    ap.add_argument("--offline-only", action="store_true",
                    help="run ONLY the offline robustness test (light on tokens)")
    args = ap.parse_args()

    llm = get_llm()
    graph = build_graph(llm, VERDICTS_DIR)
    cases = load_cases()
    if args.n:
        cases = cases[:args.n]

    if args.offline_only:
        print("\n########## OFFLINE ROBUSTNESS ONLY (search stubbed to []) ##########")
        patched = _patch_search_empty()
        try:
            final = run_case(graph, cases[0]["prompt_xy"], cases[0]["priorities"])
        finally:
            _unpatch_search(patched)
        if final is not None:
            fcs = final.get("fact_checks", []) or []
            all_unv = all(fc.get("verdict") == "unverified" for fc in fcs) if fcs else True
            print(f"\nverdict well-formed = {verdict_wellformed(final.get('verdict', {}))}")
            print(f"claims = {len(fcs)},  all marked unverified = {all_unv}")
            print("fact-check verdicts:", [fc.get("verdict") for fc in fcs])
        else:
            print("run FAILED")
        return

    runs_total = runs_ok = wellformed_ok = 0
    fc_agg = Counter()
    fc_claims_runs = 0
    order = {"consistent": 0, "evaluable": 0}
    first_bias = {"picked_first": 0, "evaluable": 0}
    per_case = []

    for c in cases:
        print(f"\n########## CASE: {c['id']} ##########")
        row = {"id": c["id"]}
        winners = {}
        for tag, prompt, first in (("xy", c["prompt_xy"], "X"), ("yx", c["prompt_yx"], "Y")):
            print(f"\n--- ordering {tag} ---")
            runs_total += 1
            final = run_case(graph, prompt, c["priorities"])
            if final is None:
                row[tag] = {"ok": False}
                continue
            runs_ok += 1
            v = final.get("verdict", {})
            wf = verdict_wellformed(v)
            wellformed_ok += 1 if wf else 0

            fs = factcheck_stats(final)
            for k in ("supported", "contradicted", "unverified"):
                fc_agg[k] += fs[k]
            fc_agg["n_claims"] += fs["n_claims"]
            fc_agg["n_non_unverified"] += fs["n_non_unverified"]
            fc_agg["n_non_unverified_with_source"] += fs["n_non_unverified_with_source"]
            fc_claims_runs += 1

            choice = classify_choice(llm, v.get("recommendation", ""), c["x"], c["y"])
            winners[tag] = choice
            if choice in ("X", "Y"):
                first_bias["evaluable"] += 1
                if choice == first:
                    first_bias["picked_first"] += 1
            row[tag] = {"ok": True, "wellformed": wf,
                        "recommendation": str(v.get("recommendation", ""))[:160],
                        "confidence": v.get("confidence"),
                        "choice": choice, "factcheck": fs,
                        "revision_count": final.get("revision_count", 0)}

        _DET = ("X", "Y", "T")   # T = honest toss-up; both saying "too close" IS consistent
        if winners.get("xy") in _DET and winners.get("yx") in _DET:
            order["evaluable"] += 1
            if winners["xy"] == winners["yx"]:
                order["consistent"] += 1
        row["order_consistent"] = (winners.get("xy") == winners.get("yx")
                                   and winners.get("xy") in _DET)
        per_case.append(row)

    # ---- offline robustness ----
    offline = None
    if not args.skip_offline and cases:
        print("\n########## OFFLINE ROBUSTNESS (search stubbed to []) ##########")
        patched = _patch_search_empty()
        try:
            c = cases[0]
            final = run_case(graph, c["prompt_xy"], c["priorities"])
        finally:
            _unpatch_search(patched)
        if final is not None:
            fcs = final.get("fact_checks", []) or []
            all_unverified = all(fc.get("verdict") == "unverified" for fc in fcs) if fcs else True
            offline = {"ok": True,
                       "wellformed": verdict_wellformed(final.get("verdict", {})),
                       "n_claims": len(fcs),
                       "all_unverified": all_unverified}
        else:
            offline = {"ok": False}

    # ---- self-consistency ----
    self_consistency = None
    if args.self_consistency and cases:
        print(f"\n########## SELF-CONSISTENCY x{args.self_consistency} ##########")
        c = cases[0]
        picks = []
        for i in range(args.self_consistency):
            print(f"\n--- repeat {i + 1}/{args.self_consistency} ---")
            final = run_case(graph, c["prompt_xy"], c["priorities"])
            if final:
                picks.append(classify_choice(llm, final["verdict"].get("recommendation", ""),
                                             c["x"], c["y"]))
        if picks:
            modal, n = Counter(picks).most_common(1)[0]
            self_consistency = {"runs": len(picks), "picks": picks,
                                "modal_agreement": round(n / len(picks), 3)}

    report = {
        "runs_total": runs_total, "runs_completed": runs_ok,
        "completion_rate": round(runs_ok / runs_total, 3) if runs_total else 0.0,
        "wellformed_rate": round(wellformed_ok / runs_ok, 3) if runs_ok else 0.0,
        "order": {**order,
                  "consistency_rate": round(order["consistent"] / order["evaluable"], 3)
                  if order["evaluable"] else None},
        "position_bias": {**first_bias,
                          "first_option_rate": round(first_bias["picked_first"] / first_bias["evaluable"], 3)
                          if first_bias["evaluable"] else None},
        "factcheck": {
            "runs": fc_claims_runs,
            "avg_claims_per_run": round(fc_agg["n_claims"] / fc_claims_runs, 2) if fc_claims_runs else 0.0,
            "supported": fc_agg["supported"], "contradicted": fc_agg["contradicted"],
            "unverified": fc_agg["unverified"],
            "pct_non_unverified_with_source":
                round(fc_agg["n_non_unverified_with_source"] / fc_agg["n_non_unverified"], 3)
                if fc_agg["n_non_unverified"] else None,
        },
        "offline_robustness": offline,
        "self_consistency": self_consistency,
        "per_case": per_case,
    }
    (HERE / "dc_eval_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print(report)
    _md(report)
    print(f"\nSaved: {HERE / 'dc_eval_results.json'} and {HERE / 'dc_eval_report.md'}")


def _pct(x):
    return "n/a" if x is None else f"{x:.1%}"


def _print(r: dict) -> None:
    print("\n===== DECISIONCOURT EVALUATION =====")
    print(f"Runs: {r['runs_completed']}/{r['runs_total']} completed  ({_pct(r['completion_rate'])})")
    print(f"Verdict well-formed rate: {_pct(r['wellformed_rate'])}")
    o = r["order"]
    print(f"\nOrder-invariance: {o['consistent']}/{o['evaluable']} dilemmas gave the same "
          f"choice in both orderings  ({_pct(o['consistency_rate'])})")
    pb = r["position_bias"]
    print(f"Position bias (picked the FIRST-mentioned option): {_pct(pb['first_option_rate'])}"
          f"  [50% = no bias]")
    fc = r["factcheck"]
    print(f"\nFact-check: {fc['avg_claims_per_run']} claims/run  "
          f"(supported={fc['supported']}, contradicted={fc['contradicted']}, unverified={fc['unverified']})")
    print(f"   non-'unverified' claims carrying a source URL: {_pct(fc['pct_non_unverified_with_source'])}")
    if r["offline_robustness"]:
        ofl = r["offline_robustness"]
        if ofl.get("ok"):
            print(f"\nOffline robustness (no search): verdict well-formed={ofl['wellformed']}, "
                  f"claims={ofl['n_claims']}, all marked unverified={ofl['all_unverified']}")
        else:
            print("\nOffline robustness: run FAILED")
    if r["self_consistency"]:
        sc = r["self_consistency"]
        print(f"\nSelf-consistency ({sc['runs']} repeats): modal agreement {_pct(sc['modal_agreement'])} "
              f"picks={sc['picks']}")


def _md(r: dict) -> None:
    o, pb, fc = r["order"], r["position_bias"], r["factcheck"]
    lines = ["# DecisionCourt — Evaluation Report", "",
             f"- Runs completed: **{r['runs_completed']}/{r['runs_total']}** ({_pct(r['completion_rate'])})",
             f"- Verdict well-formed rate: **{_pct(r['wellformed_rate'])}**",
             f"- Order-invariance (same choice in both orderings): **{_pct(o['consistency_rate'])}** "
             f"({o['consistent']}/{o['evaluable']})",
             f"- Position bias (picked first-mentioned option; 50% = unbiased): **{_pct(pb['first_option_rate'])}**",
             "", "## Fact-check coverage", "",
             f"- Avg claims/run: **{fc['avg_claims_per_run']}**",
             f"- supported: {fc['supported']}, contradicted: {fc['contradicted']}, unverified: {fc['unverified']}",
             f"- non-'unverified' claims with a source URL: **{_pct(fc['pct_non_unverified_with_source'])}**"]
    if r["offline_robustness"] and r["offline_robustness"].get("ok"):
        ofl = r["offline_robustness"]
        lines += ["", "## Offline robustness (search stubbed)", "",
                  f"- Verdict still well-formed: **{ofl['wellformed']}**",
                  f"- All {ofl['n_claims']} claims marked 'unverified': **{ofl['all_unverified']}**"]
    if r["self_consistency"]:
        sc = r["self_consistency"]
        lines += ["", "## Self-consistency", "",
                  f"- Modal agreement over {sc['runs']} repeats: **{_pct(sc['modal_agreement'])}**"]
    (HERE / "dc_eval_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
