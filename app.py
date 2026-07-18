import streamlit as st
from google import genai
from dotenv import load_dotenv
from rag import retrieve, store_document
import os

load_dotenv()
store_document()

client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

st.set_page_config(page_title="ChatGPT Clone", page_icon="🤖")
st.title("🤖 ChatGPT Clone")

if "history" not in st.session_state:
    st.session_state.history = []


for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["parts"][0]["text"])

if prompt := st.chat_input("Type your message..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.history.append({
        "role": "user",
        "parts": [{"text": prompt}]
    })

    rag_information = retrieve(prompt)

    augmented_prompt = f"""
    Use only the following information to answer.

    Context:
    {rag_information}

    Question:
    {prompt}
    """

    response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents=augmented_prompt
    )

    ai_response = response.text

    with st.chat_message("assistant"):
        st.markdown(ai_response)

    st.session_state.history.append({
        "role": "model",
        "parts": [{"text": ai_response}]
    })

