import streamlit as st
from rag import graph
from langchain_chroma import Chroma

message = st.chat_message("assistant")
message.write("Hi, what exercise would you like to know about?")

while True:
    prompt = st.chat_input()
    if prompt=='Bye':
        break
    if prompt:
        with st.chat_message("user"):
            st.write(f"{prompt}")
    
    message = st.chat_message("assistant")
    message.write(graph.invoke({'question':prompt}))
    

