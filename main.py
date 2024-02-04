from backend.core import run_llm
import streamlit as streamlit
from streamlit_chat import message
from typing import Set

streamlit.header("Documentation Helper Bot")

prompt = streamlit.text_input("Prompt", placeholder="Enter your prompt here...")

if "user_prompt_history" not in streamlit.session_state:
  streamlit.session_state["user_prompt_history"] = []

if "chat_answers_history" not in streamlit.session_state:
    streamlit.session_state["chat_answers_history"] = []

def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


if prompt:
    with streamlit.spinner("Generating response..."):
        generated_response = run_llm(query=prompt)
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )

        formatted_response = (
            f"{generated_response['result']} \n\n {create_sources_string(sources)}"
        )

        streamlit.session_state["user_prompt_history"].append(prompt)
        streamlit.session_state["chat_answers_history"].append(formatted_response)

if streamlit.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(streamlit.session_state["chat_answers_history"], streamlit.session_state["user_prompt_history"]):
        message(user_query, is_user=True)
        message(generated_response)