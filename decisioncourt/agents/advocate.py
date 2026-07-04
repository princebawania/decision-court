"""
Advocate agents (one factory, two instances: PRO and CON).

Each advocate independently (in parallel, on isolated state keys):
  1. proposes 2 web-search queries for evidence supporting its option,
  2. runs the searches (gracefully degrading to none),
  3. builds its strongest case: a persuasive argument plus up to 3
     specific FACTUAL CLAIMS that the Fact-Checker will verify.
"""

from ..llm import extract_json
from ..tools.search_tool import web_search, render_sources

QUERY_PROMPT = """You are preparing evidence to argue FOR this option:
"{option}"

Decision context: {context}

Propose 2 short, specific web search queries that could surface factual
evidence (statistics, expert findings, documented outcomes) supporting
this option.

Respond with ONLY a JSON object: {{"queries": ["q1", "q2"]}}"""

CASE_PROMPT = """You are a sharp advocate arguing FOR this option:
"{option}"

Decision being made: {decision}
Context: {context}
The decision-maker's priorities: {priorities}

WEB EVIDENCE GATHERED:
{sources}

{revision_note}
Build your strongest honest case (max 180 words). Ground it in the
evidence where possible. Then list up to 3 specific, verifiable FACTUAL
claims your case rests on (each one concrete enough to fact-check —
numbers, named findings, documented outcomes; not opinions).

Respond with ONLY a JSON object, no other text:
{{"argument": "your case", "claims": ["claim 1", "claim 2", "claim 3"]}}"""


def make_advocate_node(llm, side):
    """side: 'pro' (argues option_a) or 'con' (argues option_b)."""
    option_key = "option_a" if side == "pro" else "option_b"

    def advocate_node(state):
        option = state["framing"][option_key]
        label = side.upper()
        print(f"\n[Advocate {label}] Building case for: {option}")

        q = llm.invoke(QUERY_PROMPT.format(option=option,
                                           context=state["framing"]["context"]))
        queries = [str(x) for x in extract_json(q.content).get("queries", [])][:2]

        sources = []
        for query in queries:
            print(f"  [Advocate {label}] searching: {query}")
            sources.extend(web_search(query, max_results=3))

        resp = llm.invoke(CASE_PROMPT.format(
            option=option, decision=state["decision"],
            context=state["framing"]["context"],
            priorities=state["priorities"] or "(none given)",
            sources=render_sources(sources), revision_note=""))
        case = extract_json(resp.content)
        case = {"argument": str(case.get("argument", "")),
                "claims": [str(c) for c in case.get("claims", [])][:3]}
        print(f"  [Advocate {label}] case built, {len(case['claims'])} claim(s)")

        # ISOLATED keys only — parallel-safe partial update
        return {f"research_{side}": sources, f"case_{side}": case}
    return advocate_node


def make_revise_node(llm):
    """
    Failure-handling loop target: rebuilds any case whose claims were
    CONTRADICTED by the fact-checker. Runs once at most (revision_count).
    """
    def revise_node(state):
        updates = {"revision_count": state["revision_count"] + 1}
        for side in ("pro", "con"):
            bad = [fc for fc in state["fact_checks"]
                   if fc["side"] == side and fc["verdict"] == "contradicted"]
            if not bad:
                continue
            option = state["framing"]["option_a" if side == "pro" else "option_b"]
            note = ("IMPORTANT — a fact-checker CONTRADICTED these claims "
                    "from your previous case; you MUST drop or fix them:\n"
                    + "\n".join(f"- {fc['claim']}  (finding: {fc['note']})"
                                for fc in bad) + "\n")
            print(f"\n[Revise] Advocate {side.upper()} must fix "
                  f"{len(bad)} contradicted claim(s)")
            resp = llm.invoke(CASE_PROMPT.format(
                option=option, decision=state["decision"],
                context=state["framing"]["context"],
                priorities=state["priorities"] or "(none given)",
                sources=render_sources(state[f"research_{side}"]),
                revision_note=note))
            case = extract_json(resp.content)
            updates[f"case_{side}"] = {
                "argument": str(case.get("argument", "")),
                "claims": [str(c) for c in case.get("claims", [])][:3]}
        return updates
    return revise_node
