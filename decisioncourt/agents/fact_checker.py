"""
Fact-Checker — the trust layer between advocates and judge.

For every factual claim in both cases, it runs an independent web search
and classifies the claim as:
  supported     — evidence found that backs it
  contradicted  — evidence found that conflicts with it
  unverified    — no usable evidence found (e.g., search degraded)

Contradicted claims trigger the revision loop (once). Unverified claims
pass through but are explicitly discounted by the Judge.
"""

from ..llm import extract_json
from ..tools.search_tool import web_search, render_sources

CHECK_PROMPT = """You are a neutral fact-checker. For each claim below,
compare it against the evidence gathered for that claim and classify it.

{blocks}

Verdicts: "supported" (evidence backs it), "contradicted" (evidence
conflicts with it), "unverified" (evidence is missing or inconclusive).
Be strict: vague or unfalsifiable claims are "unverified", not supported.

Respond with ONLY a JSON object, no other text:
{{"checks": [{{"claim_id": 1, "verdict": "...", "note": "one sentence"}}, ...]}}"""


def make_fact_checker_node(llm):
    def fact_checker_node(state):
        print("\n[Fact-Checker] Verifying claims from both sides ...")
        tasks = []  # (side, claim, sources)
        for side in ("pro", "con"):
            for claim in state[f"case_{side}"]["claims"][:3]:
                print(f"  checking ({side.upper()}): {claim[:90]}")
                sources = web_search(claim, max_results=3)
                tasks.append((side, claim, sources))

        if not tasks:
            return {"fact_checks": []}

        blocks = []
        for i, (side, claim, sources) in enumerate(tasks, 1):
            blocks.append(f"CLAIM {i} (from {side.upper()} side): {claim}\n"
                          f"EVIDENCE FOR CLAIM {i}:\n{render_sources(sources)}\n")
        resp = llm.invoke(CHECK_PROMPT.format(blocks="\n".join(blocks)))
        checks = extract_json(resp.content).get("checks", [])

        by_id = {}
        for c in checks:
            try:
                by_id[int(c.get("claim_id"))] = c
            except (TypeError, ValueError):
                continue

        fact_checks = []
        for i, (side, claim, sources) in enumerate(tasks, 1):
            c = by_id.get(i, {})
            verdict = str(c.get("verdict", "unverified")).lower()
            if verdict not in ("supported", "contradicted", "unverified"):
                verdict = "unverified"
            fact_checks.append({
                "side": side, "claim": claim, "verdict": verdict,
                "note": str(c.get("note", "")),
                "sources": [s["url"] for s in sources][:3]})

        counts = {}
        for fc in fact_checks:
            counts[fc["verdict"]] = counts.get(fc["verdict"], 0) + 1
        print(f"  [Fact-Checker] verdicts: {counts}")
        return {"fact_checks": fact_checks}
    return fact_checker_node
