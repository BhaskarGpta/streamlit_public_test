import streamlit as st
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model = "llama2")

st.title("Chat with LLM")

def prompt_creation(user):
    return f"""
You are assissting with the Diagnosis of Thought methodology.
    
Start by seperating the objective truth or facts from the subjective thoughts in the provided statement.
Statement is provided here: {user}

Answer in following format: 
Objective fact:

Subjective Thought:

For the second step, provide a reasoning that both supports the subjective thought and contradicts it
Answer in following format:
Support:

Contradiction:

Now, move onto the Schema Analysis Stage. Identify any underlying schemas or patterns that might influence the approach or thoughts related to the statement.

With all this info, identify the cognitive distortion and formulate a response which is empathetic in nature and also involves some problem-focused coping
    """

user_input = st.text_input("You:", placeholder="Say whatever you are thinking...")
if user_input:
    prompt = prompt_creation(user_input)
    response = llm.invoke(prompt)
    st.write(f"LLM: {response}")