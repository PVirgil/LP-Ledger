# streamlit_app.py

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from groq import Groq
import logging

# Setup
logging.basicConfig(level=logging.INFO)
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# LLM Wrapper

def call_llm(prompt: str, model: str = "mixtral-8x7b-32768") -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a private equity fund operations analyst and LP communications officer."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Functional Modules

def generate_capital_call(df: pd.DataFrame) -> str:
    prompt = (
        f"Based on this fund cash flow schedule: {df.head(3).to_dict()}, generate a professional capital call notice."
        " Include total capital requested, deadline, wire instructions placeholder, and LP-friendly tone."
    )
    return call_llm(prompt)

def compute_fund_metrics(df: pd.DataFrame) -> str:
    prompt = (
        f"Here is fund cash flow data: {df.head(3).to_dict()}"
        " Calculate IRR, MOIC, DPI, and TVPI. Explain each metric in simple terms."
    )
    return call_llm(prompt)

def write_lp_update(df: pd.DataFrame) -> str:
    prompt = (
        f"Write a quarterly LP update letter based on this fund data: {df.head(3).to_dict()}."
        " Include portfolio company updates, key metrics, and a professional tone."
    )
    return call_llm(prompt)

def summarize_fund_doc(text: str) -> str:
    prompt = f"Summarize this fund document or clause in plain English: {text}"
    return call_llm(prompt)

def lp_qa(question: str, context: str) -> str:
    prompt = f"Fund context: {context}\nLP asks: {question}\nAnswer clearly and professionally."
    return call_llm(prompt)

# UI

def main():
    st.set_page_config("LP Ledger", page_icon="📄", layout="wide")
    st.title("📄 LP Ledger – The AI Fund Reporting Analyst")
    st.write("Upload fund data. Generate capital calls, IRR reports, LP updates, and more — instantly.")

    uploaded_file = st.file_uploader("Upload fund cash flow data (CSV)", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Fund data loaded.")
    else:
        df = pd.DataFrame()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📨 Capital Call",
        "📈 Fund Metrics",
        "📝 LP Update",
        "📁 Doc Summary",
        "💬 LP Q&A"
    ])

    with tab1:
        st.subheader("📨 Generate Capital Call")
        if st.button("Create Notice"):
            if df.empty:
                st.error("Upload data first.")
            else:
                out = generate_capital_call(df)
                st.text_area("Capital Call", value=out, height=300)

    with tab2:
        st.subheader("📈 Fund Metrics Analysis")
        if st.button("Compute Metrics"):
            if df.empty:
                st.error("Upload fund data first.")
            else:
                out = compute_fund_metrics(df)
                st.text_area("Metrics Output", value=out, height=300)

    with tab3:
        st.subheader("📝 Write LP Update")
        if st.button("Generate LP Letter"):
            if df.empty:
                st.error("Upload fund data.")
            else:
                update = write_lp_update(df)
                st.text_area("LP Update Letter", value=update, height=400)

    with tab4:
        st.subheader("📁 Fund Doc Summary")
        doc = st.text_area("Paste document or legal clause")
        if st.button("Summarize Doc"):
            if not doc:
                st.error("Paste the document.")
            else:
                summary = summarize_fund_doc(doc)
                st.text_area("Summary", value=summary, height=300)

    with tab5:
        st.subheader("💬 Ask an LP Question")
        context = st.text_area("Fund details / background")
        q = st.text_input("LP question")
        if st.button("Answer LP"):
            if not context or not q:
                st.error("Fill in both fields.")
            else:
                answer = lp_qa(q, context)
                st.markdown(f"**Response:** {answer}")

if __name__ == "__main__":
    main()
