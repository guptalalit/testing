import streamlit as st
import streamlit.components.v1 as components

# Set up the Streamlit app
st.title("File Uploader & Chat App")

# Restart session button at the top
def restart_session():
    st.session_state.messages = []
    st.rerun()

st.button("🔄 Start New Session", on_click=restart_session)

# File upload sections
st.subheader("Upload Literature Files")
literature_files = st.file_uploader("Upload literature files", type=["txt", "csv", "pdf", "jpg", "png", "xlsx", "json"], accept_multiple_files=True)
if literature_files:
    for file in literature_files:
        st.success(f"Literature file '{file.name}' uploaded successfully!")

st.subheader("Upload Session-Specific Files")
session_files = st.file_uploader("Upload session files", type=["txt", "csv", "pdf", "jpg", "png", "xlsx", "json"], accept_multiple_files=True)
if session_files:
    for file in session_files:
        st.success(f"Session file '{file.name}' uploaded successfully!")

# Chat functionality
if "messages" not in st.session_state:
    st.session_state.messages = []

st.subheader("Chat with the App")

# Improved chat UI with proper alignment and better performance
chat_html = """
<style>
.chat-container {
    max-width: 600px;
    margin: auto;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.chat-bubble {
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 70%;
    display: inline-block;
    font-size: 16px;
    word-wrap: break-word;
}
.user {
    background-color: #dcf8c6;
    align-self: flex-end;
    text-align: right;
}
.bot {
    background-color: #f1f0f0;
    align-self: flex-start;
    text-align: left;
}
</style>
<div class='chat-container'>
"""

for message in st.session_state.messages:
    role_class = "user" if message["role"] == "user" else "bot"
    chat_html += f"<div class='chat-bubble {role_class}'>{message['content']}</div>"

chat_html += "</div>"
components.html(chat_html, height=400, scrolling=True)

# Input at the bottom for faster chat updates
user_input = st.text_input("Type your message and press Enter", key="chat_input")
if st.button("Send") and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Simple response (can be replaced with AI model integration)
    bot_response = f"You said: {user_input}"  # Replace this with an actual AI response if needed
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.rerun()
