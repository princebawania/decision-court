"""
Rebuttal round — each advocate attacks the OPPONENT's case, informed by
the fact-checker's verdicts on the opponent's claims. Runs after the
fact-check gate so rebuttals target verified weaknesses, not invented ones.
"""

REBUTTAL_PROMPT = """You are the advocate FOR: "{own_option}"

Your opponent argued FOR: "{opp_option}"

OPPONENT'S ARGUMENT:
{opp_argument}

FACT-CHECK RESULTS ON THE OPPONENT'S CLAIMS:
{opp_checks}

Write a sharp rebuttal (max 120 words). Attack their weakest points —
especially claims marked contradicted or unverified. Do not repeat your
own case. Respond with the rebuttal text only, no JSON."""


def _render_checks(fact_checks, side):
    rows = [fc for fc in fact_checks if fc["side"] == side]
    if not rows:
        return "(no checkable claims)"
    return "\n".join(f"- [{fc['verdict'].upper()}] {fc['claim']} — {fc['note']}"
                     for fc in rows)


def make_rebuttal_node(llm, side):
    opp = "con" if side == "pro" else "pro"
    own_key = "option_a" if side == "pro" else "option_b"
    opp_key = "option_b" if side == "pro" else "option_a"

    def rebuttal_node(state):
        print(f"\n[Rebuttal {side.upper()}] attacking the {opp.upper()} case ...")
        resp = llm.invoke(REBUTTAL_PROMPT.format(
            own_option=state["framing"][own_key],
            opp_option=state["framing"][opp_key],
            opp_argument=state[f"case_{opp}"]["argument"],
            opp_checks=_render_checks(state["fact_checks"], opp)))
        return {f"rebuttal_{side}": resp.content.strip()}
    return rebuttal_node
