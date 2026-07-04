"""
Herald — frames the case.

Turns the user's raw dilemma into a clean two-option framing that both
advocates argue over. For yes/no questions, option_a = doing it,
option_b = not doing it.
"""

from ..llm import extract_json

PROMPT = """You are a court clerk framing a decision for adversarial review.

USER'S DECISION:
{decision}

USER'S STATED PRIORITIES (may be empty):
{priorities}

Frame this as exactly two mutually exclusive options. If it is a yes/no
decision, option_a is doing it and option_b is not doing it. Keep the
user's own wording where possible. Add one sentence of neutral context.

Respond with ONLY a JSON object, no other text:
{{"option_a": "...", "option_b": "...", "context": "one neutral sentence"}}"""


def make_herald_node(llm):
    def herald_node(state):
        print("=" * 70)
        print(f"[Herald] Framing the case: {state['decision']}")
        resp = llm.invoke(PROMPT.format(decision=state["decision"],
                                        priorities=state["priorities"] or "(none given)"))
        framing = extract_json(resp.content)
        framing = {"option_a": str(framing.get("option_a", state["decision"])),
                   "option_b": str(framing.get("option_b", "do not proceed")),
                   "context": str(framing.get("context", ""))}
        print(f"  Option A (PRO side argues): {framing['option_a']}")
        print(f"  Option B (CON side argues): {framing['option_b']}")
        return {"framing": framing}
    return herald_node
