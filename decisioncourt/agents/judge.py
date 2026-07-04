"""
Judge — the only agent that sees everything: both cases, both rebuttals,
every fact-check verdict, and the user's priorities. Scores the debate
on a fixed rubric and issues a verdict with confidence and the
conditions under which it would flip.
"""

from ..llm import extract_json

PROMPT = """You are an impartial judge deciding: {decision}

THE DECISION-MAKER'S PRIORITIES: {priorities}

OPTION A: {option_a}
CASE FOR A:
{case_pro}

OPTION B: {option_b}
CASE FOR B:
{case_con}

FACT-CHECK RESULTS (weigh these heavily):
{checks}

REBUTTAL FROM A's ADVOCATE: {rebuttal_pro}
REBUTTAL FROM B's ADVOCATE: {rebuttal_con}

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


def make_judge_node(llm):
    def judge_node(state):
        print("\n[Judge] Weighing both cases against the rubric ...")
        checks = "\n".join(
            f"- ({fc['side'].upper()}) [{fc['verdict'].upper()}] {fc['claim']} — {fc['note']}"
            for fc in state["fact_checks"]) or "(none)"
        resp = llm.invoke(PROMPT.format(
            decision=state["decision"],
            priorities=state["priorities"] or "(none given)",
            option_a=state["framing"]["option_a"],
            option_b=state["framing"]["option_b"],
            case_pro=state["case_pro"]["argument"],
            case_con=state["case_con"]["argument"],
            checks=checks,
            rebuttal_pro=state["rebuttal_pro"],
            rebuttal_con=state["rebuttal_con"]))
        v = extract_json(resp.content)
        try:
            conf = max(50, min(95, int(v.get("confidence", 60))))
        except (TypeError, ValueError):
            conf = 60
        verdict = {"recommendation": str(v.get("recommendation", "")),
                   "confidence": conf,
                   "reasoning": [str(r) for r in v.get("reasoning", [])][:4],
                   "flips_if": [str(f) for f in v.get("flips_if", [])][:3]}
        print(f"  [Judge] VERDICT: {verdict['recommendation']} "
              f"(confidence {verdict['confidence']}%)")
        return {"verdict": verdict}
    return judge_node
