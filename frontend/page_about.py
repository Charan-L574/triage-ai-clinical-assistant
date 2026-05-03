"""
pages/page_about.py — Page 7: About and Documentation.
"""
import streamlit as st

st.title("ℹ️ About TriageAI")
st.caption("Documentation, ESI reference, API setup guide, and safety disclaimer.")

st.markdown("""
> **⚠️ IMPORTANT: TriageAI is a research prototype. It has NOT been clinically validated.
> It must not replace professional medical judgment. In any emergency, call your local emergency services immediately.**
""")

st.markdown("""
## 🏥 What is TriageAI?

TriageAI is a next-generation AI-powered clinical triage assistant that takes patient symptom input and produces medically-grounded, explainable, safety-validated triage recommendations using 100% free AI APIs.

It is **not** a simple LLM wrapper. It is a multi-layer intelligence system designed to demonstrate how AI can support (not replace) emergency triage decisions.
""")

st.divider()
st.markdown("## ⚙️ How the 7-Stage Pipeline Works")

stages = [
    ("1. NLP Extraction", "Uses scispacy (or keyword fallback) to extract symptoms, body parts, conditions, medications, and procedures from the free-text input. Computes data completeness score and detects ambiguity."),
    ("2. Deterministic Risk Engine", "Hard-coded clinical rules evaluate vital sign thresholds (SpO2, heart rate, blood pressure, temperature, glucose) and keyword patterns (ACS, stroke, anaphylaxis, suicide). This stage CANNOT be overridden by any AI model."),
    ("3. RAG Disease Context", "Retrieves the 5 most semantically similar diseases from the Human Disease Ontology (14,000+ conditions) using FAISS vector search and HuggingFace embeddings."),
    ("4. LLM Reasoning (Waterfall)", "MedGemma 27B (primary) → Groq Llama-3.3 (fallback) → Gemini 2.5 Flash (fallback) → hardcoded safe escalation. Each model gets the full enriched prompt including vital signs, red flags, entities, and RAG context."),
    ("5. Schema Validation & Repair", "Validates the LLM JSON output against the Pydantic schema. Attempts automatic repair for common issues (wrong types, missing fields). Falls back to safe result if repair fails."),
    ("6. Recommendation Builder", "Maps the final ESI level to pre-written clinical recommendations. Generates targeted follow-up questions based on the differential diagnoses. Computes missing vital impact analysis."),
    ("7. Safety Validator", "The final gate. Applies risk engine overrides, confidence-based escalation, dosage stripping, and appends all mandatory safety disclaimers. Cannot be disabled."),
]
for name, desc in stages:
    with st.expander(name):
        st.write(desc)

st.divider()
st.markdown("## 📊 ESI Scale Reference")
esi_data = {
    "ESI Level": ["1 — Immediate", "2 — Emergent", "3 — Urgent", "4 — Less Urgent", "5 — Non-Urgent"],
    "Color": ["🔴 Red", "🟠 Orange", "🟡 Yellow", "🔵 Blue", "🟢 Green"],
    "Response Time": ["Immediate (seconds)", "Within 15 minutes", "Within 2–4 hours", "Within 24 hours", "Within 2–3 days"],
    "Examples": [
        "Cardiac arrest, respiratory failure, severe hemorrhage",
        "Chest pain, stroke symptoms, anaphylaxis, sepsis",
        "Abdominal pain, high fever, moderate fracture",
        "UTI, minor sprain, simple laceration",
        "Cold symptoms, medication refill, minor rash",
    ],
}
import pandas as pd
st.dataframe(pd.DataFrame(esi_data), use_container_width=True, hide_index=True)

st.divider()
st.markdown("## 🤖 Model Comparison")
model_data = {
    "Model": ["MedGemma 27B", "Groq Llama-3.3 70B", "Gemini 2.5 Flash"],
    "Role": ["Primary (medical specialist)", "Secondary fallback", "Tertiary fallback"],
    "Specialization": ["Clinical notes, EHR, biomedical lit", "General reasoning", "General reasoning"],
    "MedQA Score": ["87.7%", "~75%", "~80%"],
    "Rate Limits": ["HF free tier", "Groq free tier (generous)", "10 RPM / 250/day"],
    "Speed": ["Moderate", "Very fast (~500 tok/s)", "Fast"],
}
st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

st.divider()
st.markdown("## 🔑 Free API Keys Setup Guide")

with st.expander("1. HuggingFace Token (for MedGemma 27B)"):
    st.markdown("""
    1. Go to [huggingface.co](https://huggingface.co) and create a free account.
    2. Go to **Settings → Access Tokens** → Create a new token with **Read** permissions.
    3. **CRITICAL:** Visit [google/medgemma-27b-text-it](https://huggingface.co/google/medgemma-27b-text-it) and accept the Health AI Developer Foundation terms. Your token will NOT work until you do this.
    4. Copy your token and set `HF_TOKEN=hf_xxx...` in your `.env` file.
    """)

with st.expander("2. Groq API Key (for Llama-3.3 70B)"):
    st.markdown("""
    1. Go to [console.groq.com](https://console.groq.com) and sign in with Google/GitHub.
    2. Navigate to **API Keys** → **Create API Key**.
    3. Copy the key and set `GROQ_API_KEY=gsk_xxx...` in your `.env` file.
    4. Free tier: generous daily limits, no credit card required.
    """)

with st.expander("3. Gemini API Key (for Gemini 2.5 Flash)"):
    st.markdown("""
    1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
    2. Sign in with a Google account.
    3. Click **Create API Key** → select or create a project.
    4. Set `GEMINI_API_KEY=AIza...` in your `.env` file.
    5. Free tier: 10 requests/minute, 250 requests/day.
    """)

st.divider()
st.markdown("## ⚠️ Known Limitations")
st.markdown("""
- **Not clinically validated.** TriageAI has not been tested in any clinical environment.
- **AI can hallucinate.** LLM outputs may be plausible but incorrect. The rule engine provides a safety floor.
- **Non-English input** reduces accuracy significantly.
- **MedGemma access** requires accepting specific terms and may have queue times on HuggingFace free tier.
- **RAG index build** takes 10–20 minutes on first run and requires internet access.
- **Rate limits** may cause model fallback during high-usage periods.
- **Pediatric/geriatric** patients require specialist evaluation — the system provides a weaker guarantee for these groups.
""")

st.divider()
st.markdown("## 📚 Resources")
st.markdown("""
- [Human Disease Ontology](https://disease-ontology.org/)
- [ESI Triage Guidelines](https://www.acep.org/patient-care/policy-statements/triage-scale-standardization/)
- [MedGemma on HuggingFace](https://huggingface.co/google/medgemma-27b-text-it)
- [MedGemma Paper](https://arxiv.org/abs/2503.09560)
- [Groq Console](https://console.groq.com)
- [Google AI Studio](https://aistudio.google.com)
""")

st.divider()
st.error("""
**FULL SAFETY DISCLAIMER**

TriageAI is an artificial intelligence decision-support tool provided for research and demonstration purposes only.

It has NOT been approved by any medical regulatory authority. It has NOT been validated in clinical settings. It CANNOT and MUST NOT replace the judgment of a qualified healthcare professional.

The outputs of TriageAI are based solely on the information provided by the user and the probabilistic outputs of AI language models, which can produce errors, hallucinations, and incorrect assessments.

**If you believe you or someone else is experiencing a medical emergency, call your local emergency services immediately (112 / 999 / 911).**

By using TriageAI, you acknowledge that: (1) you will not use its outputs as a substitute for professional medical care; (2) the developers accept no liability for clinical decisions made based on its output; (3) this system is experimental and may produce incorrect results.
""")
