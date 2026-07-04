"""
Shared LangGraph state for DecisionCourt.

STATE DESIGN (decided before coding, per the orchestration checklist):

- SHARED, read-only inputs: decision, priorities, framing — written once
  by the Herald, read by everyone.
- ISOLATED per-side keys: research_pro/case_pro/rebuttal_pro vs.
  research_con/case_con/rebuttal_con. The two Advocates run in PARALLEL;
  giving each side its own keys means (a) no concurrent-write conflicts
  in LangGraph, and (b) genuine independence — neither advocate can see
  or be biased by the other's research while building its case.
- CONVERGED keys: fact_checks, verdict, memo — written after the
  parallel phase joins. The Judge is the only agent that reads everything.

Parallel-safety rule: every node returns ONLY the keys it modifies
(partial updates), never the whole state.
"""

from typing import TypedDict


class AgentState(TypedDict):
    # shared inputs (read-only after herald)
    decision: str        # the user's dilemma, verbatim
    priorities: str      # what the user says they care about ("" if none)
    framing: dict        # {"option_a": ..., "option_b": ..., "context": ...}

    # isolated: PRO side (argues for option_a)
    research_pro: list   # [{"title", "url", "snippet"}]
    case_pro: dict       # {"argument": str, "claims": [str, ...]}
    rebuttal_pro: str

    # isolated: CON side (argues for option_b)
    research_con: list
    case_con: dict
    rebuttal_con: str

    # converged (post-join)
    fact_checks: list    # [{"side", "claim", "verdict", "note", "sources"}]
    revision_count: int  # revision loop guard (max 1 revision round)
    verdict: dict        # {"recommendation", "confidence", "reasoning", "flips_if"}
    memo: str
    memo_path: str
