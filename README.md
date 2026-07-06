# ⚖️ DecisionCourt

**Live demo: https://decision-court-10.streamlit.app** — bring your own dilemma, no API key needed.

Ask ChatGPT "should I take this job?" and it'll mostly agree with whatever you
were already leaning towards. LLMs are people-pleasers. That's a bad property
in a decision advisor.

So I built the opposite: a courtroom. Two AI agents get *assigned* opposing
sides of your decision. Each one researches the web on its own and builds the
strongest case it can. A fact-checker then audits every claim they made — and
if a claim turns out to be contradicted by sources, that advocate gets sent
back to rewrite its case. Only after cross-rebuttals does a judge weigh
everything against the priorities *you* stated and hand down a verdict, with a
confidence score and the conditions under which it would flip.

The point isn't that the verdict is always right. The point is that no single
agent gets to be sycophantic, because disagreement is literally someone's job.

## Why this needs multiple agents

I tried to be honest with myself about this, since "multi-agent" is easy to
fake with one agent and extra steps. Two reasons it can't collapse into one:

1. An agent can't genuinely argue against itself. The advocates here don't
   even see each other's research while building their cases (isolated state
   keys in LangGraph), so neither can anchor on the other.
2. An agent can't audit its own claims. The fact-checker runs fresh searches
   on every claim, independent of the advocate that made it.

During one of my test runs, the PRO advocate claimed car ownership costs
"$9,122/year according to AAA" — the fact-checker searched it, found sources
saying otherwise, marked it ❌ contradicted, and forced a rewrite. Watching
that happen live is basically the whole project in one moment.

## How it's wired

Pattern: **Parallel + Aggregator**, with a fact-check gate that loops back.

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
                        | Judge |  rubric: evidence > priorities > rhetoric
                        +---+---+
                            v
                        +-------+
                        | Clerk |  writes the memo (pure code, no LLM)
                        +-------+
```

Nine nodes total. Herald frames the dilemma as two options. The advocates run
in parallel. Fact-checker classifies each claim as supported, contradicted,
or unverified. Contradicted claims trigger the revision loop — capped at one
round so the graph always terminates. Rebuttals let each side attack the
other's weakest (fact-checked) points. The judge scores it all. The clerk is
deliberately not an LLM: the final memo is assembled by plain code, so it
physically cannot misquote the verdict or the fact-check ledger.

## Things that will go wrong, and what happens when they do

This was the part of the design I spent the most time on.

- DuckDuckGo rate-limits you. Constantly. Every search call is wrapped — on
  failure it returns nothing, the advocate argues from general knowledge and
  says so, and those claims come back "unverified" instead of crashing the
  run. The judge's rubric explicitly discounts unverified claims. I tested a
  full run with search completely dead: you still get a verdict, honestly
  labelled.
- A contradicted claim after the one allowed revision? The run proceeds
  anyway and the memo discloses it. No infinite loops.
- Malformed JSON from the model gets caught by a brace-matching extractor
  with fallback defaults, so one bad judge response can't kill a trial.

I verified all of this before ever spending a real API call — the whole graph
runs against a scripted fake LLM and fake/dead search in four test scenarios.

## Running it

```bash
git clone https://github.com/princebawania/decision-court.git
cd decision-court
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # paste in a free key from console.groq.com

streamlit run app.py                 # the courtroom UI
python main.py --demo                # or the CLI version
python main.py "Buy a car or use cabs daily?" --priorities "cost, flexibility"
```

A trial takes a minute or two: around 9–13 Groq calls (Llama 3.3 70B, free
tier) plus ~10 politely rate-limited searches. Memos land in `verdicts/` —
there's a sample one committed so you can see the output format without
running anything.

## Repo layout

```
decision-court/
├── app.py                       # Streamlit courtroom UI (the live demo)
├── main.py                      # CLI entrypoint
├── decisioncourt/
│   ├── state.py                 # AgentState — the shared vs isolated split is documented here
│   ├── graph.py                 # LangGraph wiring: fan-out, join, gate, loop
│   ├── llm.py                   # Groq factory + defensive JSON parsing
│   ├── tools/search_tool.py     # DuckDuckGo with graceful degradation
│   └── agents/                  # herald, advocates, fact_checker, rebuttal, judge, clerk
├── verdicts/sample_verdict.md   # example output
└── requirements.txt
```

Built with Python, LangGraph, Groq (Llama 3.3 70B), ddgs, and Streamlit.
