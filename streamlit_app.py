import streamlit as st
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory


os.environ["GOOGLE_API_KEY"] = "AIzaSyAOatyNMmsNp0RzBm-9RjgdPHjf6MWjRuk"

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)


st.title("Chat with MindMate")

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
            Implement a conversational flow by asking follow-up questions to better understand their situation. Offer any coping techniques to help them feel better about their current state. Kep the conversation open and make them feel welcome to talk about their feelings.
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

while True:
    user_input_2 = input("How do you feel today?: ")
    user_input_2 = user_input_2.lower()

    if user_input_2 == "bye":
        print(f"Mindmate: Bye")
        break
    else:
        reply_2 = chain_2.invoke({"user_input_2": user_input_2})
    
        print(f"MindMate: {reply_2}")