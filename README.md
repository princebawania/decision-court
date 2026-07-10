# DecisionCourt — Adversarial Multi-Agent Decision Review

Bring it a hard decision. Instead of one agreeable LLM answer, you get a **structured trial**: two AI advocates independently research and argue opposing sides, a fact-checker verifies every claim they make against web sources, and a judge weighs the verified evidence against *your* priorities — delivering a verdict with a confidence level and the exact conditions that would flip it.

**Live app:** https://decision-court-10.streamlit.app/

```
$ python main.py "Should I do a Masters abroad or take my placement offer?" \
                 --priorities "career growth, finances, learning curve"
```

```
[Herald]        Option A (PRO argues): take the placement offer
                Option B (CON argues): pursue the Masters abroad
[Advocate PRO]  searching: placement offer vs masters ROI ...
[Advocate CON]  searching: MS abroad career outcomes 2026 ...
[Fact-Checker]  verdicts: {'supported': 3, 'unverified': 2, 'contradicted': 1}
[Revise]        Advocate PRO must fix 1 contradicted claim(s)
[Judge]         VERDICT: take the placement offer (confidence 71%)
[Clerk]         Decision memo saved: verdicts/verdict_20260704_183000.md
```

> **Evaluated and hardened.** I built an evaluation harness (`eval/`) that caught a real
> position bias — the judge was letting *option order* decide the verdict — and a
> fact-check degradation gap. Both are now fixed; order-invariance went from **0% to 67%**.
> See [Evaluation](#evaluation).

## The problem

Ask a single LLM "should I do X?" and it largely tells you what you want to hear — sycophancy and one-sided reasoning are documented failure modes of instruction-tuned models. People make consequential decisions (careers, money, product bets) on exactly this kind of advice.

**The engineering fix is structural, not prompt-deep:** make disagreement a job. One agent is *paid* to argue for, one *paid* to argue against, neither sees the other's research while building its case, and nothing reaches the judge without passing through an independent fact-check. A single agent cannot do this — it cannot genuinely argue against itself, and it cannot audit its own claims. That is why this system is multi-agent by necessity, not by decoration.

## Architecture

**Orchestration pattern: Parallel + Aggregator** (adversarial variant), with a **fact-check gate that forms a conditional cycle**.

```
                +---------+
                | Herald  |  frames the two options
                +----+----+
           __________|__________
          v   (parallel fan-out) v
  +--------------+      +--------------+
  | Advocate PRO |      | Advocate CON |    isolated state keys
  +------+-------+      +-------+------+
         |______________________|
                    v  (join)
             +--------------+
    +------> | Fact-Checker |  searches every claim
    |        +------+-------+
    |               v
+--------+   contradicted claims?
| Revise | <--- yes (max once)
+--------+          |
                    v  no / already revised
          Rebuttal PRO -> Rebuttal CON
                    v
                +-------+
                | Judge |  counterbalanced, rubric-scored verdict
                +---+---+
                    v
                +-------+
                | Clerk |  deterministic memo -> verdicts/
                +-------+
```

| Agent | Responsibility |
| ----- | -------------- |
| **Herald** | Frame the dilemma as two mutually exclusive options |
| **Advocate PRO / CON** | In parallel: propose search queries, gather evidence, build the strongest honest case with up to 3 verifiable claims |
| **Fact-Checker** | Independently search every claim; classify supported / contradicted / unverified; gate the revision loop. **Hard guard: a claim with no sources is forced to "unverified" — the model can't rubber-stamp it** |
| **Revise** | Rebuild any case with contradicted claims (capped at 1 round) |
| **Rebuttal PRO / CON** | Attack the opponent's case, informed by fact-check verdicts |
| **Judge** | Score both sides on a fixed rubric (evidence quality > priority fit > rebuttals; rhetoric ignored). **Counterbalanced: scores the debate in both option orders to cancel position bias** |
| **Clerk** | Deterministic (no LLM): assemble the memo — verdict, reasoning, flip conditions, full fact-check ledger with sources |

## Design decisions (the course checklist, applied)

**1. Pattern chosen upfront** — Parallel + Aggregator. The advocates are embarrassingly parallel by design; the judge is the aggregator. A supervisor pattern was rejected because no agent should "manage" the debaters — their independence is the point.

**2. Shared vs. isolated state decided before coding** — documented in `decisioncourt/state.py`. Shared read-only inputs (decision, priorities, framing); *isolated* per-side keys (`case_pro` vs `case_con`, etc.) so the parallel branches cannot conflict in LangGraph **and** cannot bias each other; converged keys (`fact_checks`, `verdict`) written only after the join. Every node returns partial updates — only the keys it owns.

**3. Failure handling built in** —

- *Revision loop:* contradicted claims send the offending advocate back to rebuild its case, capped at one round (`revision_count`), so the graph always terminates.
- *Search degradation:* every DuckDuckGo call is fenced; on rate-limit or outage it returns no sources. The fact-checker then **forces those claims to "unverified"** (deterministic guard, added after evaluation — the LLM is not trusted to do this on its own), and the judge's rubric explicitly discounts them. An offline run still delivers a verdict — honestly labelled.
- *Parse safety:* all LLM JSON passes through a fence-and-brace-matching extractor with defensive defaults; a malformed judge response cannot crash the run.

**4. Verified before the first real run, then evaluated for real** — the full graph was first exercised with a scripted fake LLM (happy path, revision-fires-once, revision-cap, total search outage). Later I built a proper evaluation harness on real dilemmas (see below), which surfaced a position bias the structural tests couldn't — and drove the two fixes now in the code.

## Trust properties

- Every claim in the memo carries a fact-check verdict and source URLs — and a claim with **no** sources is always labelled "unverified", never "supported".
- The final memo is assembled by **code, not an LLM** — the Clerk cannot misquote the verdict, the cases, or the ledger.
- The judge is **order-counterbalanced**: it evaluates both option orderings and only commits to a confident verdict when they agree; if the ordering changes the outcome, it honestly returns "too close to call" rather than letting prompt order decide.
- The judge's rubric orders evidence quality above rhetoric and confidence of tone, and must state what would flip the verdict — so the output is a decision aid, not an oracle.

## Evaluation

Structural tests prove the graph *runs*; they don't prove the verdicts are *trustworthy*. So I built an evaluation harness (`eval/`) that runs real dilemmas and measures the properties that matter for a decision aid. Each dilemma is run with its two options in **both orders** to test for position bias.

**What it measures**
- **Well-formedness** — every run yields recommendation + confidence(50-95) + reasoning + flip conditions
- **Order-invariance** — does the verdict stay the same when the option order is swapped? (a sound aid must)
- **Fact-check coverage** — avg claims/run and the share of non-"unverified" claims carrying a source URL
- **Offline robustness** — with search stubbed to nothing, does it still produce a valid verdict with all claims marked "unverified"?

**What it caught, and the fix**

The first run was humbling: **order-invariance was 0%** — the verdict flipped every single time the options were swapped, and the judge picked whichever option was mentioned *last* (classic recency bias). The offline test also showed the fact-checker calling unsourced claims "supported".

Two fixes followed: **counterbalanced two-pass judging** (score both orders, return "too close to call" on disagreement) and a **deterministic fact-check grounding guard** (no sources → "unverified").

**Results (3-dilemma set, Llama 3.3 70B)**

| Metric | Before | After |
| ------ | ------ | ----- |
| Order-invariance | 0% | **66.7%** |
| Position bias (picks first-mentioned option; 50% = unbiased) | 0% (always last) | **60%** |
| Well-formed verdicts | 100% | 100% |
| Fact-checked claims carrying a source URL | 100% | 100% |

**Run it yourself**

```
python eval/evaluate.py --n 3        # 3 dilemmas, each run in both orders
python eval/evaluate.py --offline-only   # just the search-stubbed robustness test
```
Needs `GROQ_API_KEY`. Results are written to `eval/dc_eval_results.json` and `eval/dc_eval_report.md`.

## Quickstart

Prefer not to install anything? Try the hosted app: **https://decision-court-10.streamlit.app/**


```
git clone <this repo> && cd decision-court
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# free key at https://console.groq.com — no card needed
cp .env.example .env    # paste your key into .env

python main.py --demo   # two showcase decisions
python main.py "Buy a car or use Uber daily?" --priorities "cost, flexibility"
```

A full run takes ~1-2 minutes: fast Groq LLM calls (Llama 3.3 70B) plus a handful of polite DuckDuckGo searches.

## Project structure

```
decision-court/
├── main.py                      # CLI entrypoint (+ --demo)
├── decisioncourt/
│   ├── state.py                 # AgentState — shared/isolated design documented
│   ├── graph.py                 # LangGraph wiring: parallel fan-out, gate, cycle
│   ├── llm.py                   # Groq factory + robust JSON extraction
│   ├── tools/search_tool.py     # DuckDuckGo search with graceful degradation
│   └── agents/
│       ├── herald.py            # framing
│       ├── advocate.py          # PRO/CON case-builders + revise node
│       ├── fact_checker.py      # claim verification + no-source guard
│       ├── rebuttal.py          # cross-examination round
│       ├── judge.py             # counterbalanced, rubric-scored verdict
│       └── clerk.py             # deterministic memo writer
├── eval/                        # evaluation harness
│   ├── evaluate.py              # order-invariance, fact-check, offline robustness
│   ├── decision_cases.json      # labelled dilemmas (both orderings)
│   └── dc_eval_report.md        # generated results
├── verdicts/                    # generated decision memos land here
├── requirements.txt
└── .env.example
```

## Tech stack

Python · LangGraph · LangChain · Groq (Llama 3.3 70B) · DuckDuckGo Search (ddgs)
