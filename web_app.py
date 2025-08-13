import streamlit as st
from chatbot_engine import generate_response

st.set_page_config(page_title="SR Chatbot", layout="centered")

st.title("💬 SR AI Chatbot")
st.markdown("Your personal assistant for motivation, beauty, and growth 💄✨")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("You:", key="input")

if user_input:
    reply = generate_response(user_input)
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("SR", reply))

# Display chat history

for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(f"<div style='text-align: right; background-color: #dcf8c6; padding: 8px; border-radius: 10px; margin: 5px;'>**{speaker}:** {message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='text-align: left; background-color: #f1f0f0; padding: 8px; border-radius: 10px; margin: 5px;'>**{speaker}:** {message}</div>", unsafe_allow_html=True)