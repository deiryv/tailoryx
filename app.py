# @packages
from dataclasses import dataclass
from typing import Literal
import json
import streamlit as st
import urllib.request

# Page configuration
st.set_page_config(page_title="Coca-Cola KMA")


@dataclass
class Message:
    """
    Class to contain & track messages
    """

    origin: Literal["human", "AI"]
    message: str


def load_css():
    """
    Gets page styles
    """
    with open("./static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)


def initialize_session_state():
    """
    Creates session state for convo with the LLM
    """
    # Define chat history
    if "history" not in st.session_state:
        st.session_state.history = []


def convert_history_to_json(history):
    """
    Returns chat history in a format required by the Prompt Flow endpoint.
    Accepts the LLM convo 'history' stored in state containing a list of Message objects.
    """
    refactored_history = []

    # Iterate Message objects
    idx = 0
    while idx < len(history):
        convo_dict = {"inputs": {"query": {}}, "outputs": {"reply": {}}}
        # If current Message is from the user
        if history[idx].origin == "human":
            # Store query in the required format
            convo_dict["inputs"]["query"] = history[idx].message
            # Then move to the next Message to process the AI response
            idx += 1
            if idx < len(history) and history[idx].origin == "AI":
                convo_dict["outputs"]["reply"] = history[idx].message
            # Store results
            refactored_history.append(convo_dict)
        # Skip to next message
        idx += 1

    # Display comparison
    print("\nHistory: ", history)
    print("\nRefactored Chat History: ", refactored_history)

    return refactored_history


def call_promptflow(query):
    """
    Returns a response from the Prompt Flow endpoint.
    Accepts the human_prompt as a query for the LLM.
    """
    # Prompt Flow endpoint URL, associated API key, the deployment name
    ENDPOINT_URL = st.secrets["OPEN_AI_ENDPOINT"]
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

    # Retreive and convert chat history
    #chat_history_list = convert_history_to_json(st.session_state.history)

    # The parameters requested by the flow
    data = {
    "model": "gpt-3.5-turbo",  # Specify the model
    "messages": [
        {"role": "user", "content": query}  # Include the user's query as a message
    ]
}
    body = str.encode(json.dumps(data))

    # Provide the OpenAI API key in the header
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}",
    }

    # Set the OpenAI API endpoint
    ENDPOINT_URL = "https://api.openai.com/v1/chat/completions"

    # Create the request
    req = urllib.request.Request(ENDPOINT_URL, body, headers)

    try:
      
        with urllib.request.urlopen(req) as response:
            response_data = response.read()
            # Parse the JSON response
            response_json = json.loads(response_data)

            # Extract the answer from the response
            answer = response_json.get("choices")[0].get("message").get("content")
            print(response_json)
            # Output the answer (this can also be stored in the session state or displayed in Streamlit)
            #st.write("Response:", answer)
            print(answer)
            return answer

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # The request ID and the timestamp for debugging
        print(error.info())
        print(error.read().decode("utf8", "ignore"))


def get_filepaths(llm_response):
    """
    Returns filepaths for documents utilized in the LLM response.
    Accepts the llm_response from the Prompt Flow endpoint.
    """
    filepaths = []
    # Filter by score for filepaths
    for obj in llm_response["documents"]:
        if obj["score"] > 0.0152:
            filepaths.append(obj["filepath"])
    # Remove duplicates
    unique_filepaths = list(set(filepaths))

    return unique_filepaths


def on_click_callback():
    """
    Manages chat history in session state and calls the LLM
    """
    # Get the prompt from session state
    human_prompt = st.session_state.human_prompt

    # Call the defined LLM endpoint
    llm_response = call_promptflow(human_prompt)

    # Get related file names & format response
    #filepaths = get_filepaths(llm_response)
    llm_answer = llm_response
    # Persist prompt and llm_response in session state
    st.session_state.history.append(Message("human", human_prompt))
    st.session_state.history.append(Message("AI", llm_answer))

    # Clear prompt value
    st.session_state.human_prompt = ""


def main():
    load_css()
    initialize_session_state()

    # Setup web page text
    st.image("./static/logo.png", width=400)
    #st.markdown(Ta, unsafe_allow_html=True)
    st.markdown(
        "<h3>Hello 💬</h3>",
        unsafe_allow_html=True,
    )

    # Create a container for the chat between the user & LLM
    chat_placeholder = st.container()
    # Create a form for the user prompt
    prompt_placeholder = st.form("chat-form")

    # Display chat history within chat_placehoder
    with chat_placeholder:
        for chat in st.session_state.history:
            div = f"""
                <div class="chat-row {'' if chat.origin == 'AI' else 'row-reverse'}">
                <img class="chat-icon" src="{'https://ask-chatbot.netlify.app/public/ai_icon.png' if chat.origin == 'AI' else 'https://ask-chatbot.netlify.app/public/user_icon.png'}" width=32 height=32>
                <div class="chat-bubble {'ai-bubble' if chat.origin == 'AI' else 'human-bubble'}">&#8203;{chat.message}</div>
                </div>
                """
            st.markdown(div, unsafe_allow_html=True)

            for _ in range(3):
                st.markdown("")

    # Create the user prompt within prompt_placeholder
    with prompt_placeholder:
        st.markdown("**Chat**")
        cols = st.columns((6, 1))
        cols[0].text_input(
            "Chat",
            placeholder="Send a message",
            label_visibility="collapsed",
            key="human_prompt",
        )
        cols[1].form_submit_button(
            "Submit",
            type="primary",
            on_click=on_click_callback,
        )


if __name__ == "__main__":
    main()
