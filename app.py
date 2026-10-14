"""FinNews Insight: English financial-news intelligence for Streamlit Cloud."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from html import escape
from pathlib import Path

import streamlit as st

from intelligence import article_body
from model_runtime import ModelRuntime, run_intelligence
from model_delivery import ensure_model
from news_utils import extract_file

ROOT = Path(__file__).resolve().parent
APP_VERSION = "2.2"
logger = logging.getLogger("finnews")
logging.basicConfig(level=logging.INFO)
st.set_page_config(page_title="FinNews Insight", layout="wide", initial_sidebar_state="collapsed")


def setting(name, default=None):
    """Environment settings for Colab/local use; Streamlit secrets for Cloud."""
    if os.getenv(name):
        return os.environ[name]
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    return default


def model_default():
    configured_path = setting("SENTIMENT_MODEL_PATH")
    if configured_path:
        return configured_path
    configured = setting("SENTIMENT_MODEL")
    if configured:
        return configured
    for folder in ("finbert-finetuned", "distilbert_ep3", "finnews-distilbert"):
        if (ROOT / "models" / folder / "config.json").exists():
            return f"models/{folder}"
    # The default is the same student model published in the project Release.
    return "__github_release_model__"


def resolve_model(value):
    p = Path(value).expanduser()
    for candidate in ([p, ROOT / p] if not p.is_absolute() else [p]):
        if candidate.is_dir():
            if not (candidate / "config.json").is_file():
                raise ValueError("The model folder does not contain config.json.")
            return str(candidate.resolve())
    if p.is_absolute() or value.replace("\\", "/").startswith("models/"):
        raise FileNotFoundError("Set SENTIMENT_MODEL to the published fine-tuned model repository or a valid local model folder.")
    return value.strip()


@st.cache_resource(show_spinner=False)
def model_runtime():
    return ModelRuntime()


def analyze_article(text, model_source):
    article = article_body(text)
    if len(article.strip()) < 30:
        raise ValueError("Please upload an article containing at least a few complete sentences.")
    token = setting("HF_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
    models = {
        "sentiment": model_source,
        "entity": setting("ENTITY_MODEL", "dslim/distilbert-NER"),
        "summary": setting("SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6"),
    }
    status = st.empty()
    status.info("Preparing your news insights. Please keep this page open.")
    try:
        if model_source == "__github_release_model__":
            model_source = ensure_model()
            models["sentiment"] = model_source
        result = run_intelligence(article, model_runtime(), models, progress=status.info)
        result["diagnostics"]["version"] = APP_VERSION
        logger.info("FINNEWS_DIAGNOSTICS %s", json.dumps(result["diagnostics"]))
        return result
    finally:
        status.empty()


def render_article_text(value):
    # Escape untrusted uploaded text and bypass Markdown math syntax for dollar amounts.
    st.markdown(f'<div class="fn-body">{escape(value)}</div>', unsafe_allow_html=True)


st.markdown("<style>" + (ROOT / "styles.css").read_text() + "</style>", unsafe_allow_html=True)
st.markdown('<div class="fn-brand">FINNEWS INSIGHT</div>', unsafe_allow_html=True)
st.title("Understand the news. Know the impact.")
st.markdown('<div class="fn-intro">Financial news, distilled into perspective — company intelligence, sentiment signals, and the context that matters.</div>', unsafe_allow_html=True)

with st.container(border=True):
    st.subheader("Upload your news")
    uploaded = st.file_uploader("Choose an English news document", type=["txt", "md", "docx", "pdf"],
                                help="Supported: TXT, DOCX, Markdown and PDFs with a selectable text layer.")
    st.caption("English financial news · TXT / DOCX / text-based PDF · Up to 10 MB. Scanned PDFs require OCR first.")
    text = ""
    if uploaded is not None:
        try:
            text = extract_file(uploaded.name, uploaded.getvalue())
            st.success("Your document is ready to analyze.")
        except Exception as exc:
            st.error(str(exc))
    clicked = st.button("Click to get insights", type="primary", disabled=not bool(text))

model_id = model_default()
summary_id = setting("SUMMARY_MODEL", "sshleifer/distilbart-cnn-12-6")
entity_id = setting("ENTITY_MODEL", "dslim/distilbert-NER")
signature = hashlib.sha256((text + model_id + summary_id + entity_id + APP_VERSION).encode()).hexdigest()
if st.session_state.get("signature") != signature:
    st.session_state.pop("analysis", None)

if clicked:
    try:
        source = ensure_model() if model_id == "__github_release_model__" else resolve_model(model_id)
        result = analyze_article(text, source)
        result["source"] = uploaded.name
        st.session_state["analysis"] = result
        st.session_state["signature"] = signature
    except Exception:
        logger.exception("Article analysis failed")
        st.error("We could not analyze this article. Please try a readable English news file. If the issue continues, ask the app owner to check the service configuration and server log.")

result = st.session_state.get("analysis")
if result:
    st.divider()
    st.header("Your news insights")
    for i, row in enumerate(result["companies"]):
        with st.container(border=True):
            left, right = st.columns([2, 1])
            with left:
                label = "Main company mentioned" if i == 0 else "Also mentioned"
                st.markdown(f'<div class="fn-label">{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="fn-company">{escape(row["name"])}</div>', unsafe_allow_html=True)
            with right:
                sentiment = row["sentiment"]
                title = "Unclear" if sentiment == "unclear" else sentiment.capitalize()
                st.markdown('<div class="fn-label">Financial sentiment</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="fn-sentiment fn-{sentiment}">{title}</span>', unsafe_allow_html=True)
            st.caption(row["note"])
            if row.get("evidence"):
                with st.expander(f"Why this result for {row['name']}?"):
                    for sentence in row["evidence"]:
                        render_article_text(sentence)
    if not result["companies"]:
        st.info("No company could be identified reliably from this article. We have not guessed a company name.")
        st.write("**Overall article sentiment:**", result["overall"]["sentiment"].capitalize())
    with st.container(border=True):
        st.subheader("News summary")
        render_article_text(result["summary"])
        if result["summary_method"] == "extractive":
            st.caption("Summary uses selected sentences from the article because generated summarization was unavailable.")
        else:
            st.caption("Automatically generated. Check the original article for important figures and context.")
    clipped = result["entity_coverage"]["truncated"] or result["summary_coverage"]["truncated"]
    clipped = clipped or any(row.get("coverage", {}).get("truncated", False) for row in result["companies"])
    if result.get("overall"):
        clipped = clipped or result["overall"].get("coverage", {}).get("truncated", False)
    if clipped:
        st.warning("This document is long. Some insights cover only part of the article; use a shorter article for full coverage.")
    if result["entity_coverage"]["company_limit"]:
        st.caption("Only the most frequently mentioned organizations are shown.")
    with st.expander("View original text"):
        st.text(text[:20000])
        if len(text) > 20000:
            st.caption("Preview shortened. Refer to your original document for the complete text.")
    st.caption("Company names are detected from organization mentions and may need review. For multi-company news, sentiment uses attributable sentences, not a dedicated company-impact model.")

st.divider()
st.caption("Research assistance only. Financial sentiment is not a prediction of share-price movements or investment advice.")
