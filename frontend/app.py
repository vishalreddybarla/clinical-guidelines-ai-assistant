"""Streamlit chat interface for the Clinical Guidelines AI Assistant."""

from __future__ import annotations

import os

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api")

DISCLAIMER = (
    "**Disclaimer:** This tool is for informational and educational purposes only. "
    "It is NOT a substitute for professional clinical judgment. Always consult "
    "current guidelines and apply clinical expertise when making patient care decisions."
)

GUIDELINE_SOURCES = [
    "Any",
    "World Health Organization",
    "American Diabetes Association",
    "American Heart Association / American College of Cardiology",
    "Centers for Disease Control and Prevention",
    "U.S. Preventive Services Task Force",
]

MODELS = {
    "GPT-4o Mini (fast, cheap)": "gpt-4o-mini",
    "GPT-4o (best quality)": "gpt-4o",
    "Claude 3.5 Haiku (fast)": "claude-3-5-haiku-20241022",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clinical Guidelines AI Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🏥 Clinical Guidelines AI")
    st.markdown("**Powered by RAG + Claude/GPT**")
    st.divider()

    selected_source = st.selectbox("Filter by guideline source", GUIDELINE_SOURCES)
    selected_model_label = st.selectbox("LLM Model", list(MODELS.keys()))
    selected_model = MODELS[selected_model_label]

    st.divider()
    st.subheader("Available Guidelines")
    st.markdown(
        """
- 🩺 WHO Hypertension (2021)
- 💉 ADA Diabetes (2024)
- ❤️ AHA Heart Failure (2022)
- 🦠 CDC STI Treatment (2021)
- 🔬 USPSTF Cancer Screening
        """
    )

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    # Health check widget
    with st.expander("API Status"):
        try:
            health = requests.get(f"{API_BASE}/health", timeout=3).json()
            st.success(f"✅ {health.get('status', 'unknown')}")
            st.write(f"Guidelines loaded: **{health.get('guidelines_loaded', 0)}**")
            st.write(f"Total chunks: **{health.get('total_chunks', 0)}**")
        except Exception:
            st.error("❌ API unreachable")

    st.divider()
    st.caption(DISCLAIMER)

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
st.title("🏥 Clinical Guidelines AI Assistant")
st.markdown(
    "Ask clinical questions grounded in **WHO, ADA, AHA, CDC, and USPSTF** guidelines."
)
st.warning(DISCLAIMER)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query_id" not in st.session_state:
    st.session_state.last_query_id = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Example questions
if not st.session_state.messages:
    st.subheader("Try one of these example questions:")
    examples = [
        "What is the first-line treatment for Stage 1 hypertension?",
        "What is the target HbA1c for most adults with Type 2 diabetes?",
        "At what age should breast cancer screening begin for average-risk women?",
        "Are there interactions between warfarin and amoxicillin?",
        "What are the stages of heart failure classification according to AHA?",
    ]
    cols = st.columns(len(examples))
    for col, example in zip(cols, examples):
        with col:
            if st.button(example, use_container_width=True, key=f"ex_{example[:20]}"):
                st.session_state.prefill_question = example
                st.rerun()

# Handle pre-filled question from example buttons
if "prefill_question" in st.session_state:
    prompt = st.session_state.pop("prefill_question")
else:
    prompt = st.chat_input("Ask a clinical question…")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching guidelines…"):
            try:
                payload = {
                    "query": prompt,
                    "model": selected_model,
                    "prompt_version": "v3_few_shot",
                }
                if selected_source != "Any":
                    payload["filter_source"] = selected_source

                resp = requests.post(f"{API_BASE}/ask", json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()

                answer = data["answer"]
                st.markdown(answer)

                # Citations expander
                if data.get("citations"):
                    with st.expander(f"📚 Sources ({len(data['citations'])} retrieved)", expanded=True):
                        for i, cite in enumerate(data["citations"], 1):
                            st.markdown(
                                f"**{i}. {cite['source_organization']}** — "
                                f"{cite['guideline_title']}, Page {cite['page_number']} "
                                f"_(relevance: {cite['relevance_score']:.2f})_"
                            )

                # Drug interactions
                if data.get("drug_interactions"):
                    with st.expander("⚠️ Drug Interaction Alert", expanded=True):
                        for interaction in data["drug_interactions"]:
                            severity = interaction.get("severity", "Unknown")
                            color = "🔴" if "major" in severity.lower() else "🟡"
                            st.warning(
                                f"{color} **{severity}**: {interaction['description']}"
                            )

                # PubMed articles
                if data.get("pubmed_articles"):
                    with st.expander("🔬 Recent Research (PubMed)"):
                        for art in data["pubmed_articles"]:
                            st.markdown(
                                f"- [{art['title']}]({art['url']})  \n"
                                f"  *{art['journal']}, {art['pub_date']}*"
                            )

                # Metadata row
                cols = st.columns(4)
                cols[0].metric("Model", data["model_used"].split("/")[-1])
                cols[1].metric("Tokens", data["tokens_used"])
                cols[2].metric("Latency", f"{data['latency_ms']} ms")
                cols[3].metric("Est. Cost", f"${data['cost_estimate_usd']:.5f}")

                st.session_state.last_query_id = data.get("query_id")
                st.session_state.last_response = data
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except requests.HTTPError as exc:
                err = f"API error: {exc.response.status_code} — {exc.response.text}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except requests.ConnectionError:
                err = (
                    "Cannot reach the API. Make sure the FastAPI backend is running:\n\n"
                    "```bash\nuvicorn api.main:app --reload\n```"
                )
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as exc:
                err = f"Unexpected error: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

# Feedback section (only shown after a response exists)
if st.session_state.last_query_id:
    st.divider()
    st.subheader("Rate this answer")
    rating = st.feedback("stars")
    feedback_comment = st.text_input(
        "Optional comment", placeholder="What could be improved?", key="feedback_comment"
    )
    if st.button("Submit Feedback"):
        try:
            resp = requests.post(
                f"{API_BASE}/feedback",
                json={
                    "query_id": st.session_state.last_query_id,
                    "rating": (rating or 0) + 1,
                    "comment": feedback_comment or None,
                },
                timeout=10,
            )
            if resp.ok:
                st.success("Thank you for your feedback!")
                st.session_state.last_query_id = None
            else:
                st.error("Failed to submit feedback.")
        except Exception as exc:
            st.error(f"Error submitting feedback: {exc}")
