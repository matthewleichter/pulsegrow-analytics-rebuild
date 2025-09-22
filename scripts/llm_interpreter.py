import streamlit as st
from utils.llm_interpreter_utils import interpret_results

def run_llm_interpreter():
    st.title("LLM Model Output Interpreter")
    input_text = st.text_area("Paste model output:")
    if input_text:
        interpretation = interpret_results(input_text)
        st.write("Interpretation:", interpretation)
