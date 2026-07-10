# DecisionCourt — Evaluation (add to your repo)

Adds a real, defensible evaluation to DecisionCourt. A debate/verdict system
has no single "right answer", so this measures trustworthiness properties, not
accuracy against a key.

## What it measures
- **Well-formedness** — every run yields recommendation + confidence(50-95) + reasoning + flip conditions
- **Order-invariance (position bias)** — each dilemma is run with the two options in BOTH orders;
  a sound aid should pick the same option regardless of which is argued first
- **Fact-check coverage** — avg claims/run and the share of non-"unverified" claims that carry a source URL
- **Offline robustness** — with web search stubbed to [], it should still produce a valid verdict
  with all claims marked "unverified" (validates the graceful-degradation design; no network)
- **Self-consistency** (optional, `--self-consistency N`) — run one dilemma N times, measure agreement

## Setup
Copy this folder into your repo as `eval/`:
```
decision-court/
├── eval/
│   ├── evaluate.py
│   ├── decision_cases.json
│   └── README_DC_EVAL.md
├── decisioncourt/ ...
└── main.py
```
Make sure `GROQ_API_KEY` is in your `.env` (same one the app uses).

## Run it
```bash
python eval/evaluate.py                # all 6 cases (12 runs; ~15-25 min)
python eval/evaluate.py --n 3          # first 3 cases only (faster)
python eval/evaluate.py --self-consistency 3   # add a stability test
python eval/evaluate.py --skip-offline # skip the search-stubbed test
```
Each full run makes ~9-13 Groq calls + a few DuckDuckGo searches (Groq free tier
is generous; if DDG rate-limits, the system degrades and claims become 'unverified',
which the eval reports honestly). Outputs: dc_eval_results.json + dc_eval_report.md.

## What to send me
Paste the printed summary (or dc_eval_report.md). I need: completion rate,
well-formed rate, order-invariance %, position-bias %, fact-check coverage,
and the offline-robustness line.

---

## Resume bullets (fill [X] after you run it)

**One-page:**
- Built DecisionCourt, an adversarial multi-agent decision-review system (LangGraph):
  two advocates argue opposing sides, a fact-checker verifies claims via web search,
  and a judge issues a rubric-scored verdict
- Evaluated it on a dilemma test set: **[X]% well-formed verdicts**, **[X]% order-invariance**
  (same choice regardless of which option is argued first), and graceful offline degradation
  with 100% of unverifiable claims correctly flagged

**Two-page (add):**
- Measured position bias by running each dilemma in both option orderings; fact-checked
  claims carried source URLs in **[X]%** of non-'unverified' cases

> Interpreting the numbers:
> - order-invariance high (e.g. >80%) = the adversarial design resists position bias — a strong claim
> - position-bias "first-option rate" near 50% = unbiased; far from 50% = the system favors
>   whichever option is mentioned first (worth mentioning honestly + as future work)
