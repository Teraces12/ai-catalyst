import streamlit as st
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
access_code = os.getenv("ACCESS_CODE", "terasecret123")  # Set your code in .env

# Simple session-based access control
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Access Required")
    user_code = st.text_input("Enter Access Code", type="password")
    if st.button("Submit"):
        if user_code == access_code:
            st.success("✅ Access granted")
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("❌ Invalid access code")
    st.stop()

# ---- Main App Starts Here ----
st.set_page_config(
    page_title="AI Catalyst PDF Assistant",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.title("AI Catalyst PDF Assistant 🧠")
st.subheader("Summarize or ask questions from your PDF using LangChain + OpenAI")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
mode = st.radio("What do you want to do?", ["Summarize", "Ask a question"])

# Advanced options
model_name = st.selectbox("Select model:", ["gpt-3.5-turbo-16k", "gpt-4"])
temperature = st.slider("Temperature (creativity):", 0.0, 1.0, 0.0, step=0.1)
allow_non_english = st.checkbox("Allow non-English PDFs", value=False)

col1, col2 = st.columns(2)
with col1:
    start_page = st.number_input("Start Page", min_value=1, value=1)
with col2:
    end_page = st.number_input("End Page", min_value=start_page, value=start_page + 4)

question = ""
if mode == "Ask a question":
    question = st.text_input("Enter your question:", placeholder="e.g. What is the core mission of this document?")

if uploaded_file and (mode == "Summarize" or (mode == "Ask a question" and question)):
    with st.spinner("Processing..."):
        progress = st.progress(0)
        try:
            endpoint = "/summarize" if mode == "Summarize" else "/ask"
            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
            data = {
                "model_name": model_name,
                "temperature": temperature,
                "allow_non_english": str(allow_non_english).lower(),
                "start_page": int(start_page),
                "end_page": int(end_page)
            }

            if mode == "Ask a question":
                data["question"] = question

            response = requests.post(f"{backend_url}{endpoint}", files=files, data=data)
            response.raise_for_status()
            result = response.json()

            answer = result.get("answer")
            language = result.get("language", "unknown")
            citations = result.get("citations", [])

            if answer:
                st.success(answer)
                st.info(f"🌍 Detected Language: {language}")
                if citations:
                    st.caption(f"📚 Citations from: {', '.join(citations)}")
            else:
                st.error("Unexpected response format.")

        except requests.exceptions.HTTPError as http_err:
            try:
                error_msg = response.json().get("error", str(http_err))
                st.error(f"Error: {error_msg}")
            except:
                st.error(f"HTTP error: {http_err}")
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            progress.progress(100)
