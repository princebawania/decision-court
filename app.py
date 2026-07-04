"""
DecisionCourt — Streamlit courtroom UI.

Run locally:   streamlit run app.py
The same LangGraph backend as main.py, streamed node-by-node into a live
courtroom view: opposing cases side by side, a fact-check ledger with
verdict badges, revision alerts, and a final verdict card.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from decisioncourt.graph import build_graph, initial_state
from decisioncourt.llm import get_llm

load_dotenv()

BADGE = {"supported": "✅ supported",
         "contradicted": "❌ contradicted",
         "unverified": "❓ unverified"}

EXAMPLES = [
    "Should I do a Masters abroad or take my placement offer?",
    "Should a small D2C brand spend on Instagram influencers or Google ads?",
    "Buy a car or use cabs daily?",
    "Should our startup build our own ML model or use an API?",
]

st.set_page_config(page_title="DecisionCourt", page_icon="⚖️", layout="wide")

st.title("⚖️ DecisionCourt")
st.caption("Adversarial multi-agent decision review — two AI advocates argue "
           "opposing sides, a fact-checker audits every claim against web "
           "sources, and a judge delivers a verdict weighted to *your* priorities.")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Groq API key", type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Free key at console.groq.com — no card needed.")
    st.divider()
    st.subheader("Try an example")
    example = st.radio("Examples", EXAMPLES, index=None,
                       label_visibility="collapsed")
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Herald frames your dilemma as two options\n"
        "2. Two advocates research & argue — in parallel, isolated\n"
        "3. Fact-checker verifies every claim (contradicted claims "
        "force a rewrite)\n"
        "4. Cross-rebuttals\n"
        "5. Judge scores evidence > rhetoric and issues the verdict")

decision = st.text_area("**The decision you're facing**",
                        value=example or "",
                        placeholder="e.g. Should I switch jobs now or wait for my bonus?",
                        height=80)
priorities = st.text_input("**What matters to you** (optional)",
                           placeholder="e.g. career growth, low risk, family time")

go = st.button("🔨  Hold the trial", type="primary",
               disabled=not decision.strip())

if go:
    if not api_key:
        st.error("Add your Groq API key in the sidebar first (free at console.groq.com).")
        st.stop()
    os.environ["GROQ_API_KEY"] = api_key

    llm = get_llm()
    graph = build_graph(llm, verdicts_dir="verdicts")
    state = dict(initial_state(decision.strip(), priorities.strip()))

    status = st.status("⚖️ Court is in session ...", expanded=True)
    framing_ph = st.empty()
    col_pro, col_con = st.columns(2)
    pro_ph = col_pro.empty()
    con_ph = col_con.empty()
    ledger_ph = st.empty()
    rebuttal_ph = st.empty()
    verdict_ph = st.empty()

    def render_case(ph, side_label, option, case, checks):
        with ph.container(border=True):
            st.subheader(f"{side_label}: {option}")
            if case.get("argument"):
                st.write(case["argument"])
            if checks:
                st.markdown("**Claims:**")
                for fc in checks:
                    st.markdown(f"- {BADGE[fc['verdict']]} — {fc['claim']}")

    def checks_for(side):
        return [fc for fc in state.get("fact_checks", []) if fc["side"] == side]

    try:
        for update in graph.stream(state, config={"recursion_limit": 50},
                                   stream_mode="updates"):
            for node, delta in update.items():
                state.update(delta or {})

                if node == "herald":
                    status.write("📜 **Herald** framed the case.")
                    f = state["framing"]
                    framing_ph.info(f"**Option A:** {f['option_a']}  \n"
                                    f"**Option B:** {f['option_b']}  \n"
                                    f"*{f['context']}*")
                elif node == "advocate_pro":
                    status.write("🟢 **Advocate PRO** finished researching and built its case.")
                    render_case(pro_ph, "🟢 The case FOR",
                                state["framing"]["option_a"],
                                state["case_pro"], [])
                elif node == "advocate_con":
                    status.write("🔴 **Advocate CON** finished researching and built its case.")
                    render_case(con_ph, "🔴 The case FOR",
                                state["framing"]["option_b"],
                                state["case_con"], [])
                elif node == "fact_checker":
                    counts = {}
                    for fc in state["fact_checks"]:
                        counts[fc["verdict"]] = counts.get(fc["verdict"], 0) + 1
                    status.write(f"🔍 **Fact-Checker** audited every claim: {counts}")
                    render_case(pro_ph, "🟢 The case FOR",
                                state["framing"]["option_a"],
                                state["case_pro"], checks_for("pro"))
                    render_case(con_ph, "🔴 The case FOR",
                                state["framing"]["option_b"],
                                state["case_con"], checks_for("con"))
                elif node == "revise":
                    status.write("⚠️ **Revision ordered** — contradicted claims "
                                 "sent back to the offending advocate.")
                    st.toast("A contradicted claim was caught — case being rewritten!",
                             icon="⚠️")
                elif node == "rebuttal_pro":
                    status.write("🟢 **PRO** delivered its rebuttal.")
                elif node == "rebuttal_con":
                    status.write("🔴 **CON** delivered its rebuttal.")
                    with rebuttal_ph.container(border=True):
                        st.subheader("⚔️ Cross-examination")
                        c1, c2 = st.columns(2)
                        c1.markdown("**🟢 PRO attacks CON's case:**")
                        c1.write(state["rebuttal_pro"])
                        c2.markdown("**🔴 CON attacks PRO's case:**")
                        c2.write(state["rebuttal_con"])
                elif node == "judge":
                    status.write("👨‍⚖️ **Judge** has reached a verdict.")
                elif node == "clerk":
                    status.write("📁 **Clerk** filed the decision memo.")

        status.update(label="✅ Trial complete", state="complete", expanded=False)

        v = state["verdict"]
        with verdict_ph.container(border=True):
            st.subheader("👨‍⚖️ The Verdict")
            m1, m2 = st.columns([3, 1])
            m1.markdown(f"### {v['recommendation']}")
            m2.metric("Confidence", f"{v['confidence']}%")
            st.progress(v["confidence"] / 100.0)
            st.markdown("**Why:**")
            for i, r in enumerate(v["reasoning"], 1):
                st.markdown(f"{i}. {r}")
            st.markdown("**This verdict flips if:**")
            for f in v["flips_if"]:
                st.markdown(f"- {f}")
            if state.get("revision_count"):
                st.caption(f"⚠️ {state['revision_count']} revision round was "
                           "triggered after the fact-checker caught contradicted claims.")
            if state.get("memo"):
                st.download_button("⬇️ Download the full decision memo (markdown)",
                                   state["memo"],
                                   file_name="decision_memo.md")

    except Exception as e:
        status.update(label="Trial interrupted", state="error")
        st.error(f"Something went wrong mid-trial: {e}")
        st.info("Most common cause: Groq rate limit — wait ~30s and retry. "
                "Search outages are handled automatically (claims become "
                "'unverified'), so they never crash a run.")
