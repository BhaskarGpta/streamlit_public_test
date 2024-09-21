import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import base64

os.environ["GOOGLE_API_KEY"] = "AIzaSyAOatyNMmsNp0RzBm-9RjgdPHjf6MWjRuk"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

icon_path = "image.png"

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are assisting with the Diagnosis of Thought methodology.

            Conversation history:
            {history}

            Start by separating the objective truth or facts from the subjective thoughts in the user-provided statement.
            Do not mention this in the response

            For the second step, provide reasoning that both supports the subjective thought and contradicts it.
            Answer in the following way:
            "You are feeling this way because (supportive reasoning), but (contradictive reasoning)."

            Now, move on to the Schema Analysis stage. Identify any underlying schemas or patterns that might influence the approach or thoughts related to the statement.
            Do not mention this in the response

            Using all this information, identify the cognitive distortion.

            Now, formulate an empathetic response, comforting the user, based on all the information you have gathered.
            Implement a conversational flow by asking follow-up questions to better understand their situation. Offer any coping techniques to help them feel better about their current state. Keep the conversation open and make them feel welcome to talk about their feelings.
            Do not mention the objective thought, subjective thought, schema analysis, or cognitive distortion in your response.
            """,
        ),
        ("human", "{user_input_2}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

chain_2 = LLMChain(
    prompt=prompt,
    llm=llm,
    memory=memory
)

st.set_page_config(page_title="Chat with MindMate", layout="wide")

st.markdown("""
    <style>
    .user-bubble {
        background-color: #4BBF9D; /* Softer teal */
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 30px;
        width: fit-content;
        float: right;
        clear: both;
    }
    .bot-bubble {
        background-color: #FF7F7F; /* Softer red */
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 30px;
        width: fit-content;
        float: left;
        clear: both;
    }
    </style>
    """, unsafe_allow_html=True)


if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

st.title("Chat with MindMate")

user_input_2 = st.text_input("How do you feel today?", "")

if st.button("Send"):
    if user_input_2:
        st.session_state.conversation_history.append({"user": user_input_2})

        try:
            reply_2 = chain_2.invoke({"user_input_2": user_input_2.lower()})

            st.session_state.conversation_history.append({"bot": reply_2['text']})
        except Exception as e:
            st.session_state.conversation_history.append({"bot": f"Error: {e}"})

for message in st.session_state.conversation_history:
    if "user" in message:
        st.markdown(f'<div class="user-bubble">{message["user"]}</div>', unsafe_allow_html=True)
    else:
        with open(icon_path, "rb") as icon_file:
            icon_data = base64.b64encode(icon_file.read()).decode()
            st.markdown(f'''
            <div style="float: left; margin-bottom: 5px;">
                <img src="data:image/jpeg;base64,{icon_data}" style="width: 80px; height: 80px;">
            </div>
            <div class="bot-bubble">
                {message["bot"]}
        ''', unsafe_allow_html=True)

