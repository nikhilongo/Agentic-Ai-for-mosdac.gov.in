import streamlit as st
from rag import graph  # assumes graph is already compiled in my_graph.py

st.set_page_config(page_title="LangGraph Runner", layout="centered")
st.title("🚀 LangGraph App")

# Input section
user_input = st.text_input("Enter your input:", "")

if st.button("Run Graph"):
    if not user_input.strip():
        st.warning("Please enter some input.")
    else:
        with st.spinner("Running LangGraph..."):
            try:
                # If your graph expects {"input": user_input}, adjust as needed
                output = graph.invoke({"input": user_input})
                st.success("Graph execution complete.")
                st.write("### Output:")
                st.json(output)
            except Exception as e:
                st.error(f"Graph execution failed: {e}")
