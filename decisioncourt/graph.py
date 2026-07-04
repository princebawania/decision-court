"""
LangGraph wiring for DecisionCourt.

ORCHESTRATION PATTERN: Parallel + Aggregator, with an adversarial twist
and a fact-check gate (conditional cycle).

                        +---------+
                        | herald  |   frames the two options
                        +----+----+
                   __________|__________
                  v (parallel fan-out)  v
          +--------------+      +--------------+
          | advocate_pro |      | advocate_con |     isolated state keys
          +------+-------+      +-------+------+
                 |____________ _________|
                              v  (join)
                       +--------------+
              +------> | fact_checker |  verifies every claim via search
              |        +------+-------+
              |               |
   (contradicted claims       | route
    & no revision yet)        v
        +--------+     all clear / already revised
        | revise | <---+      |
        +--------+            v
                       rebuttal_pro -> rebuttal_con
                              |
                              v
                          +-------+
                          | judge |   rubric-scored verdict
                          +---+---+
                              v
                          +-------+
                          | clerk |   deterministic memo -> verdicts/
                          +---+---+
                              v
                             END

FAILURE HANDLING (checklist item 3):
- revision loop: contradicted claims send the offending advocate back
  to rebuild its case — capped at 1 round (revision_count guard), so
  the graph always terminates
- web search degrades to [] on any error; claims then become
  "unverified" and the judge explicitly discounts them
- every LLM JSON parse is fenced by extract_json + defensive defaults
"""

from langgraph.graph import StateGraph, END

from .state import AgentState
from .agents.herald import make_herald_node
from .agents.advocate import make_advocate_node, make_revise_node
from .agents.fact_checker import make_fact_checker_node
from .agents.rebuttal import make_rebuttal_node
from .agents.judge import make_judge_node
from .agents.clerk import make_clerk_node


def route_after_fact_check(state):
    contradicted = any(fc["verdict"] == "contradicted"
                       for fc in state["fact_checks"])
    if contradicted and state["revision_count"] < 1:
        return "revise"
    return "rebuttal_pro"


def build_graph(llm, verdicts_dir="verdicts"):
    g = StateGraph(AgentState)
    g.add_node("herald", make_herald_node(llm))
    g.add_node("advocate_pro", make_advocate_node(llm, "pro"))
    g.add_node("advocate_con", make_advocate_node(llm, "con"))
    g.add_node("fact_checker", make_fact_checker_node(llm))
    g.add_node("revise", make_revise_node(llm))
    g.add_node("rebuttal_pro", make_rebuttal_node(llm, "pro"))
    g.add_node("rebuttal_con", make_rebuttal_node(llm, "con"))
    g.add_node("judge", make_judge_node(llm))
    g.add_node("clerk", make_clerk_node(verdicts_dir))

    g.set_entry_point("herald")
    # parallel fan-out: both advocates run concurrently on isolated keys
    g.add_edge("herald", "advocate_pro")
    g.add_edge("herald", "advocate_con")
    # join: fact_checker waits for both
    g.add_edge("advocate_pro", "fact_checker")
    g.add_edge("advocate_con", "fact_checker")
    # fact-check gate: revise (once) or proceed to rebuttals
    g.add_conditional_edges("fact_checker", route_after_fact_check,
                            {"revise": "revise",
                             "rebuttal_pro": "rebuttal_pro"})
    g.add_edge("revise", "fact_checker")
    g.add_edge("rebuttal_pro", "rebuttal_con")
    g.add_edge("rebuttal_con", "judge")
    g.add_edge("judge", "clerk")
    g.add_edge("clerk", END)
    return g.compile()


def initial_state(decision, priorities=""):
    return {
        "decision": decision, "priorities": priorities,
        "framing": {}, "research_pro": [], "case_pro": {},
        "rebuttal_pro": "", "research_con": [], "case_con": {},
        "rebuttal_con": "", "fact_checks": [], "revision_count": 0,
        "verdict": {}, "memo": "", "memo_path": "",
    }
