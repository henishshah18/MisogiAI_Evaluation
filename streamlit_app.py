import streamlit as st
from rag import graph
from langchain_chroma import Chroma


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.chat_message("assistant"):
    st.write("Hi, what exercise would you like to know about?")
# message.write("Hi, what exercise would you like to know about?")

prompt = st.chat_input("Enter query here")
if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
else:
    st.write("Please enter some text.")


response  = graph.invoke({'question':prompt})
with st.chat_message("assistant"):
    st.markdown(response['answer'])
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response['answer']})
    

