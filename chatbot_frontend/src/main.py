import os
import requests
import streamlit as st

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/salt-rag-agent")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about the SALT Call for Proposals document.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )

    st.header("Example Questions")
    st.markdown("- When is the deadline for proposals 1 and 2?")
    st.markdown("- What is the wavelength range covered by NIRWALS?")
    st.markdown("- Describe RSS instrument and its modes' availability")
    # st.markdown(
    #     "- At which hospitals are patients complaining about billing and "
    #     "insurance issues?"
    # )
    # st.markdown("- What is the average duration in days for closed emergency visits?")
    # st.markdown(
    #     "- What are patients saying about the nursing staff at "
    #     "Castaneda-Hardy?"
    # )
    # st.markdown("- What was the total billing amount charged to each payer for 2023?")
    # st.markdown("- What is the average billing amount for medicaid visits?")
    # st.markdown("- Which physician has the lowest average visit duration in days?")
    # st.markdown("- How much was billed for patient 789's stay?")
    # st.markdown(
    #     "- Which state had the largest percent increase in medicaid visits "
    #     "from 2022 to 2023?"
    # )
    # st.markdown("- What is the average billing amount per day for Aetna patients?")
    # st.markdown("- How many reviews have been written from patients in Florida?")
    # st.markdown(
    #     "- For visits that are not missing chief complaints, "
    #     "what percentage have reviews?"
    # )
    # st.markdown(
    #     "- What is the percentage of visits that have reviews for each hospital?"
    # )
    # st.markdown(
    #     "- Which physician has received the most reviews for this visits "
    #     "they've attended?"
    # )
    # st.markdown("- What is the ID for physician James Cooper?")
    # st.markdown(
    #     "- List every review for visits treated by physician 270. Don't leave any out."
    # )

st.title("SALT Chatbot")
st.info(
    "Ask me questions about SALT Call for proposals document"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        headers={
            'Content-Type': 'application/json'
        }
        response = requests.get(CHATBOT_URL, json=data, headers=headers)

        if response.status_code == 200:
            output_text = response.json()["answer"]
            explanation = response.json()["intermediate_steps"]

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )