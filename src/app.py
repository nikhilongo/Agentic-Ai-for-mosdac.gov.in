import streamlit as st
import os
import sys

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.graph import graph
from src.core.vector_store import vector_store_manager

# Page Config
st.set_page_config(page_title="MOSDAC Chatbot", layout="centered")
st.title("🚀 MOSDAC Chatbot")

# Initialize Vector Store (Ingest data if needed)
# In a real app, this might be done separately or checked more robustly
if not os.path.exists("./chroma_isro"):
    with st.spinner("Initializing Knowledge Base..."):
        vector_store_manager.ingest_data()
        st.success("Knowledge Base Initialized!")

# Input section
user_input = st.text_input("Enter your query:", "what is tempreture in jaipur?")
st.caption("(this can be slow, uploaded on free cloud services)")

if st.button("Run"):
    if not user_input.strip():
        st.warning("Please enter some input.")
    else:
        with st.spinner("Processing..."):
            try:
                # The graph expects a list of messages or a dictionary with "messages"
                output = graph.invoke({"messages": [("human", user_input)]})
                
                # Extract the final response
                # The output contains the state, so we look at the last message
                last_message = output["messages"][-1]
                st.success("Done!")
                st.write("### Response:")
                st.write(last_message.content)
                
                # Optional: Show full debug info
                with st.expander("Debug Info"):
                    st.json(output)
                    
            except Exception as e:
                st.error(f"Execution failed: {e}")
