"""Interactive demo: map a clinical term, inspect candidates and confidence.

    streamlit run app.py
"""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from termmap import SYSTEMS
from termmap.config import Config
from termmap.engine import TermMapEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

NAMES = {"ICD10CM": "ICD-10-CM", "SNOMEDCT_US": "SNOMED CT"}
EXAMPLES = {
    "OCR noise": ("chronickidney disease stag 3", ""),
    "RA, rheumatology": ("ra", "morning stiffness in hands lasting 2 hours, positive rheumatoid factor, on methotrexate"),
    "RA, echo report": ("ra", "echo: dilated ra and rv, tricuspid regurgitation"),
    "MS, neurology": ("ms", "relapsing remitting course, new periventricular lesions on mri brain, started on ocrelizumab"),
    "MS, ward note": ("ms", "patient confused, altered, oriented to self only"),
    "Unspecified diabetes": ("diabetes", ""),
    "Specific diabetes": ("type 2 diabetes with diabetic neuropathy", "numbness in both feet, a1c 8.2"),
}

st.set_page_config(page_title="TermMap", layout="wide")


@st.cache_resource(show_spinner="Loading dictionaries and building indexes (about a minute on first start)...")
def load_engine() -> TermMapEngine:
    cfg = Config.load()
    path = Path(cfg.artifact_dir) / "engine.joblib"
    engine = TermMapEngine.load(cfg, path)
    for s in SYSTEMS:
        engine.mapper(s)
    return engine


def map_terms(engine: TermMapEngine, df: pd.DataFrame) -> dict:
    df = df.reset_index(drop=True)
    return {s: engine.adjudicate(s, df, engine.mapper(s).predict(df)) for s in SYSTEMS}


def pair_signal(engine: TermMapEngine, icd: str, snomed: str) -> str:
    cw = getattr(engine, "crosswalk", None)
    if cw is None:
        return "unknown"
    seen = cw.by_snomed.get(snomed)
    if seen and icd in seen:
        return "agree"
    if seen or cw.by_icd.get(icd):
        return "disagree"
    return "no evidence"


SIGNAL_TEXT = {
    "agree": "The two predictions form a pair the coding team has used before. "
             "On held-out data, ICD-10-CM accuracy for agreeing pairs is about 89%.",
    "disagree": "The coding team has paired these codes with other codes. "
                "On held-out data, ICD-10-CM accuracy for such terms is about 32%: a candidate for review.",
    "no evidence": "No pairing evidence for these codes in the labelled data.",
    "unknown": "",
}

st.title("TermMap")
st.caption("Clinical term mapping to ICD-10-CM and SNOMED CT")

engine = load_engine()
tab_single, tab_batch = st.tabs(["Map a term", "Map a file"])

with tab_single:
    cols = st.columns(len(EXAMPLES))
    for col, (label, (term, ctx)) in zip(cols, EXAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state["term"], st.session_state["ctx"] = term, ctx

    term = st.text_input("Clinical term", key="term", placeholder="e.g. afib w rvr")
    ctx = st.text_area("Chart context (optional)", key="ctx", height=90,
                       placeholder="Surrounding text from the chart chunk")
    sub = st.selectbox("Entity category (optional)",
                       ["", "DiagnosisCode", "ProcedureCode", "LabService", "EncounterType", "Device"])

    if term:
        df = pd.DataFrame({"entity": [term], "chunk_text": [ctx]})
        if sub:
            df["subconcept"] = [sub]
        t = time.time()
        preds = map_terms(engine, df)
        elapsed = time.time() - t

        left, right = st.columns(2)
        for col, system in zip((left, right), SYSTEMS):
            p = preds[system][0]
            title = engine.mapper(system).term.title(p.code)
            with col:
                st.subheader(NAMES[system])
                st.metric(label=title or "no candidate", value=p.code or "-", delta=f"confidence {p.confidence:.2f}",
                          delta_color="off")
                top = pd.DataFrame(
                    [(c, engine.mapper(system).term.title(c), round(s, 3)) for c, s in p.ranked[:5]],
                    columns=["code", "description", "score"],
                )
                st.dataframe(top, hide_index=True, use_container_width=True)

        signal = pair_signal(engine, preds["ICD10CM"][0].code, preds["SNOMEDCT_US"][0].code)
        st.markdown(f"**Cross-system check: {signal}.** {SIGNAL_TEXT[signal]}")
        st.caption(f"Mapped in {elapsed:.2f} s")

with tab_batch:
    st.write("CSV with a `clinical_entity` (or `entity`) column. `chunk_id`, `chunk_text` and `subconcept` are used when present.")
    up = st.file_uploader("Input file", type=["csv"])
    if up is not None:
        df = pd.read_csv(up, dtype=str, keep_default_na=False)
        if "clinical_entity" in df.columns and "entity" not in df.columns:
            df = df.rename(columns={"clinical_entity": "entity"})
        if "chunk_text" not in df.columns:
            df["chunk_text"] = ""
        with st.spinner(f"Mapping {len(df)} terms..."):
            out = engine.predict(df)
        out = out.rename(columns={"entity": "clinical_entity"})
        keep = [c for c in ("chunk_id", "clinical_entity", "icd10cm_code", "snomed_code",
                            "icd10cm_code_confidence", "snomed_code_confidence") if c in out.columns]
        st.dataframe(out[keep], hide_index=True, use_container_width=True)
        conf = out[["icd10cm_code_confidence", "snomed_code_confidence"]].min(axis=1)
        st.write(f"{(conf < 0.3).sum()} of {len(out)} terms have a confidence below 0.3 in at least one system.")
        buf = io.StringIO()
        out[keep].to_csv(buf, index=False)
        st.download_button("Download predictions", buf.getvalue(), "predictions.csv", "text/csv")
