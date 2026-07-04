"""
Clerk — deterministic (no LLM). Assembles the final decision memo from
the structured state and saves it. Because this is code, not a model,
the memo can never misquote the verdict, the cases, or the fact-checks.
"""

import os
from datetime import datetime

ICON = {"supported": "[OK]", "contradicted": "[X]", "unverified": "[?]"}


def make_clerk_node(verdicts_dir):
    def clerk_node(state):
        v = state["verdict"]
        lines = [
            f"# Decision Memo: {state['decision']}",
            "",
            f"**Verdict: {v['recommendation']}**",
            f"**Confidence: {v['confidence']}%**",
            "",
            "## Why",
        ]
        lines += [f"{i}. {r}" for i, r in enumerate(v["reasoning"], 1)]
        lines += ["", "## This verdict flips if"]
        lines += [f"- {f}" for f in v["flips_if"]] or ["- (none identified)"]

        lines += ["", "---", "", "## The case that was argued", "",
                  f"### For: {state['framing']['option_a']}",
                  state["case_pro"]["argument"], "",
                  f"**Rebuttal from the other side:** {state['rebuttal_con']}",
                  "",
                  f"### For: {state['framing']['option_b']}",
                  state["case_con"]["argument"], "",
                  f"**Rebuttal from the other side:** {state['rebuttal_pro']}"]

        lines += ["", "## Fact-check ledger", ""]
        if state["fact_checks"]:
            for fc in state["fact_checks"]:
                lines.append(f"- {ICON[fc['verdict']]} ({fc['side'].upper()}) "
                             f"{fc['claim']} — *{fc['note']}*")
                for u in fc["sources"]:
                    if u:
                        lines.append(f"    - {u}")
        else:
            lines.append("- No checkable claims were made.")

        if state["revision_count"]:
            lines += ["", f"*Note: {state['revision_count']} revision round "
                          "was triggered after contradicted claims.*"]

        memo = "\n".join(lines)
        os.makedirs(verdicts_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(verdicts_dir, f"verdict_{ts}.md")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(memo)
        print(f"\n[Clerk] Decision memo saved: {path}")
        return {"memo": memo, "memo_path": path}
    return clerk_node
