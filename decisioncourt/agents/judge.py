"""
Judge — the only agent that sees everything: both cases, both rebuttals,
every fact-check verdict, and the user's priorities. Scores the debate
on a fixed rubric and issues a verdict with confidence and the
conditions under which it would flip.

DEBIASED (added after evaluation): the evaluation found the judge suffered
from position/recency bias — it favored whichever option appeared LAST in
the prompt, so the verdict flipped whenever option order was swapped
(0% order-invariance). Fix: COUNTERBALANCED TWO-PASS judging. We score the
debate twice — once with A presented first, once with B presented first —
and:
  * if both passes pick the SAME option -> confident verdict (avg confidence)
  * if they DISAGREE -> the result is order-sensitive, so we honestly report
    a "too close to call" verdict with low confidence instead of letting
    prompt order decide the outcome.
"""

import re

from ..llm import extract_json

PROMPT = """You are an impartial judge deciding: {decision}

THE DECISION-MAKER'S PRIORITIES: {priorities}

OPTION A: {option_a}
CASE FOR A:
{case_a}

OPTION B: {option_b}
CASE FOR B:
{case_b}

FACT-CHECK RESULTS (weigh these heavily):
{checks}

REBUTTAL FROM A's ADVOCATE: {rebuttal_a}
REBUTTAL FROM B's ADVOCATE: {rebuttal_b}

Scoring rubric — apply in this order:
1. Evidence quality: arguments resting on SUPPORTED claims outweigh
   those on unverified ones. CONTRADICTED claims count against a side.
2. Fit to the decision-maker's stated priorities.
3. Strength of rebuttals: did either side land unanswered blows?
4. Ignore rhetoric and confidence of tone entirely.

Respond with ONLY a JSON object, no other text:
{{"recommendation": "the option you recommend, stated plainly",
  "confidence": <integer 50-95>,
  "reasoning": ["reason 1", "reason 2", "reason 3"],
  "flips_if": ["condition that would flip this verdict", "another"]}}"""


def _tokens(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _which_option(recommendation, option_a, option_b):
    """Map a free-text recommendation onto 'a' or 'b' by token overlap."""
    r = _tokens(recommendation)
    a = len(r & _tokens(option_a))
    b = len(r & _tokens(option_b))
    if a > b:
        return "a"
    if b > a:
        return "b"
    return "?"


def _conf(v, default=60):
    try:
        return max(50, min(95, int(v.get("confidence", default))))
    except (TypeError, ValueError):
        return default


def make_judge_node(llm):
    def _score(decision, priorities, opt_a, case_a, opt_b, case_b,
               rebut_a, rebut_b, checks):
        resp = llm.invoke(PROMPT.format(
            decision=decision, priorities=priorities or "(none given)",
            option_a=opt_a, case_a=case_a, option_b=opt_b, case_b=case_b,
            checks=checks, rebuttal_a=rebut_a, rebuttal_b=rebut_b))
        return extract_json(resp.content)

    def judge_node(state):
        print("\n[Judge] Weighing both cases (counterbalanced, 2 passes) ...")
        checks = "\n".join(
            f"- ({fc['side'].upper()}) [{fc['verdict'].upper()}] {fc['claim']} — {fc['note']}"
            for fc in state["fact_checks"]) or "(none)"

        opt_a = state["framing"]["option_a"]   # argued by PRO
        opt_b = state["framing"]["option_b"]   # argued by CON
        case_a = state["case_pro"]["argument"]
        case_b = state["case_con"]["argument"]
        reb_a = state["rebuttal_pro"]          # PRO attacks CON
        reb_b = state["rebuttal_con"]          # CON attacks PRO

        # Pass 1: real A first.  Pass 2: real B first (everything swapped).
        v1 = _score(state["decision"], state["priorities"],
                    opt_a, case_a, opt_b, case_b, reb_a, reb_b, checks)
        v2 = _score(state["decision"], state["priorities"],
                    opt_b, case_b, opt_a, case_a, reb_b, reb_a, checks)

        pick1 = _which_option(v1.get("recommendation", ""), opt_a, opt_b)
        pick2 = _which_option(v2.get("recommendation", ""), opt_a, opt_b)
        c1, c2 = _conf(v1), _conf(v2)

        if pick1 in ("a", "b") and pick1 == pick2:
            # Both orderings agree -> robust, confident verdict.
            verdict = {
                "recommendation": str(v1.get("recommendation", "")),
                "confidence": round((c1 + c2) / 2),
                "reasoning": [str(r) for r in v1.get("reasoning", [])][:4],
                "flips_if": [str(f) for f in v1.get("flips_if", [])][:3],
            }
        elif pick1 in ("a", "b") and pick2 in ("a", "b") and pick1 != pick2:
            # Order-sensitive -> honest toss-up, don't let prompt order decide.
            reasoning = (["The two sides are closely matched: the verdict changed "
                          "when the option order was swapped, so the arguments are "
                          "not decisive on their own."]
                         + [str(r) for r in v1.get("reasoning", [])][:1]
                         + [str(r) for r in v2.get("reasoning", [])][:1])[:4]
            flips_if = ([str(f) for f in v1.get("flips_if", [])][:1]
                        + [str(f) for f in v2.get("flips_if", [])][:1]) or \
                       ["clearer evidence favouring one option"]
            verdict = {
                "recommendation": (f"Too close to call between \"{opt_a}\" and "
                                   f"\"{opt_b}\" — decide on your own weighting of "
                                   "the priorities below"),
                "confidence": min(c1, c2, 55),
                "reasoning": reasoning,
                "flips_if": flips_if[:3],
            }
        else:
            # Ambiguous mapping -> fall back to pass 1 (can't confirm agreement).
            verdict = {
                "recommendation": str(v1.get("recommendation", "")),
                "confidence": c1,
                "reasoning": [str(r) for r in v1.get("reasoning", [])][:4],
                "flips_if": [str(f) for f in v1.get("flips_if", [])][:3],
            }

        print(f"  [Judge] VERDICT: {verdict['recommendation']} "
              f"(confidence {verdict['confidence']}%)")
        return {"verdict": verdict}
    return judge_node
