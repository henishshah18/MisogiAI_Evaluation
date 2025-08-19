import streamlit as st
from rag import graph
from langchain_chroma import Chroma


with st.chat_message("assistant"):
    st.write("Hi, what exercise would you like to know about?")
# message.write("Hi, what exercise would you like to know about?")
prompt = st.chat_input("Enter query here")
if prompt:
    with st.chat_message("user"):
        st.write(f"{prompt}")
    
    message = st.chat_message("assistant")
    response  = graph.invoke({'question':prompt})
    message.write(response["answer"])
    

