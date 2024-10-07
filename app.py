# @packages
from dataclasses import dataclass
from typing import Literal
import json
import streamlit as st
import urllib.request
import pandas as pd
from openai import OpenAI
import time

# Page configuration
st.set_page_config(page_title="TailoryX")
dataset = []
OPEN_AI_ENDPOINT = st.secrets["OPEN_AI_ENDPOINT"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
ASSISTANT_ID = st.secrets["ASSISTANT_ID"]

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
assistant = client.beta.assistants.retrieve(st.secrets["ASSISTANT_ID"])

@dataclass
class Message:
    """
    Class to contain & track messages
    """

    origin: Literal["human", "AI", "dataframe", "bar", "line"]
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
    # Initialise session state variables
    if "file_uploaded" not in st.session_state:
        st.session_state.file_uploaded = False

    if "assistant_text" not in st.session_state:
        st.session_state.assistant_text = [""]

    if "code_input" not in st.session_state:
        st.session_state.code_input = []

    if "code_output" not in st.session_state:
        st.session_state.code_output = []

    if "disabled" not in st.session_state:
        st.session_state.disabled = False

    if "file" not in st.session_state:
        st.session_state.file = ""


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

    # Create the request
    req = urllib.request.Request(OPEN_AI_ENDPOINT, body, headers)

    try:
      
        with urllib.request.urlopen(req) as response:
            response_data = response.read()
            # Parse the JSON response
            response_json = json.loads(response_data)

            # Extract the answer from the response
            answer = response_json.get("choices")[0].get("message").get("content")
            #print(response_json)
            # Output the answer (this can also be stored in the session state or displayed in Streamlit)
            #st.write("Response:", answer)
            #print(answer)
            return answer

    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        # The request ID and the timestamp for debugging
        print(error.info())
        print(error.read().decode("utf8", "ignore"))

def on_click_callback():
    """
    Manages chat history in session state and calls the LLM
    """
    human_prompt = st.session_state.human_prompt
    st.session_state.history.append(Message("human", human_prompt))

    # Call the defined LLM endpoint
    llm_response = generate_prompt()
    write_response(llm_response)
    #print(f"----- LLM Response {llm_response}")

    # Persist prompt and llm_response in session state
    
    #st.session_state.history.append(Message("AI", llm_response))

    # Clear prompt value
    st.session_state.human_prompt = ""

    # Delete the uploaded file from OpenAI
    client.files.delete(st.session_state.file.id)


def load_document():
    with st.sidebar:
        st.image("./static/logo.png", width=200)
    # File uploader for CSV
        uploaded_file = st.file_uploader(":computer: Load a CSV file:", type="csv")

        if uploaded_file:
            # Read the uploaded CSV file
            file_name = "./data/"+uploaded_file.name
            # Initialize the datasets dictionary
            dataset = pd.read_csv(uploaded_file)
            dataset.to_csv(path_or_buf=file_name)
            file = client.files.create(
                file=open(file_name, "rb"),
                purpose='assistants'
            )
            # Display the name and contents of the file
            #st.write(f"File uploaded: {file_name}")
            st.session_state.file = file
                        # Verifica si ya existe un dataframe en el historial
            dataframe_in_history = False
            for message in st.session_state.history:
                if message.origin == "dataframe":
                    # Si ya existe, actualiza el dataframe
                    message.message = dataset
                    dataframe_in_history = True
                    break

            if not dataframe_in_history:
                # Si no existe, añade el dataframe al historial
                st.session_state.history.append(Message("dataframe", dataset))
    #return datasets

def write_response(response_str: str):
    """
    Write a response from an agent to a Streamlit app.

    Args:
        response_dict: The response from the agent.

    Returns:
        None.
    """

      # Parse the JSON string into a Python dictionary
    try:
        response_dict = json.loads(response_str)
    except json.JSONDecodeError as e:
        st.error(f"Failed to parse JSON: {e}")
        return

    # Check if the response has the main 'answer'.
    if "answer" in response_dict:
        answer = response_dict["answer"]
        st.session_state.history.append(Message("AI", answer))

    # Check if there is additional information.
    if "additional_info" in response_dict and "answer" in response_dict["additional_info"]:
        additional_answer = response_dict["additional_info"]["answer"]
        st.session_state.history.append(Message("AI", additional_answer))

    # Check if the response is a bar chart.
    if "bar" in response_dict:
        data = response_dict["bar"]
        df = pd.DataFrame(data["data"], index=data["columns"], columns=["Revenue"])
        st.session_state.history.append(Message("bar", df))
        # st.bar_chart(df)

    # Check if the response is a line chart.
    if "line" in response_dict:
        data = response_dict["line"]
        df = pd.DataFrame(data["data"], index=data["columns"], columns=["Value"])
        st.session_state.history.append(Message("line", df))
        st.line_chart(df)

    # Check if the response is a table.
    if "table" in response_dict:
        data = response_dict["table"]
        df = pd.DataFrame(data["data"], columns=data["columns"])
        st.table(df)


def create_assistant():
    # Initialise the OpenAI client, and retrieve the assistant
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    assistant = client.beta.assistants.retrieve(st.secrets["ASSISTANT_ID"])
    return client, assistant


def generate_prompt():
    if  st.session_state.file is not None:
        with st.status("Starting work...", expanded=False) as status_box:
            human_prompt = st.session_state.human_prompt
            print(f"Human prompt: {human_prompt}")
            print(f"File {st.session_state.file.id}")
            # Create a new thread with a message that has the uploaded file's ID
            thread = client.beta.threads.create(
                messages=[
                    {
                    "role": "user",
                    "content": human_prompt,
                    # Attach the new file to the message.
                    "attachments": [
                        { "file_id": st.session_state.file.id, "tools": [{"type": "code_interpreter"}] }
                    ],
                    }
                    ]
                )
            # The thread now has a vector store with that file in its tool resources.
            print(thread.tool_resources.code_interpreter)
            print(f"Thread {thread}")

            # Create a run with the new thread
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id,
            )

            #print(f"---- Run {run}")

            # Check periodically whether the run is done, and update the status
            import time

            start_time = time.time() 
            timeout = 120 

            while run.status != "completed":
                elapsed_time = time.time() - start_time 

                if elapsed_time >= timeout: 
                    print("Tiempo excedido. Cancelando la ejecución...")
                    client.beta.threads.runs.cancel(run.id, thread_id=thread.id)
                    break  

                time.sleep(5)
                print(run.status)
                status_box.update(label=f"{run.status}...", state="running")

                run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)


            # Once the run is complete, update the status box and show the content
            status_box.update(label="Complete", state="complete", expanded=True)
            # Fetch all messages in the thread
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )

            #print(f"Complete response from Assistant: {messages}")
            #messages.data[0].content[0].text.value

            # Initialize an empty list to store the texts
            filtered_texts = []

            # Iterate through the messages to filter those with an assistant_id and extract their text
            for message in messages.data:
                #print(f"---- message: {message}")
                # Only consider messages where assistant_id is not None
                if message.assistant_id is not None:
                    # Each message can have multiple content blocks
                    for content_block in message.content:
                        if content_block.type == 'text':  # Ensure we are handling text blocks
                            # Append the text value to the list
                            filtered_texts.append(content_block.text.value)

            # Combine all the texts into one string
            all_texts = " ".join(filtered_texts)

            # Print the complete response
            print(f" ----- Complete All text from Assistant: {all_texts}")

            # Return the concatenated text
            return all_texts
def main():
    load_css()
    initialize_session_state()
    #assistant = create_assistant()
    load_document()

    # Create a container for the chat between the user & LLM
    chat_placeholder = st.container()
    # Create a form for the user prompt
    prompt_placeholder = st.form("chat-form")

    message = ""
    chatrow = ""
    chaticon = ""

    # Display chat history within chat_placehoder
    with chat_placeholder:
        for chat in st.session_state.history:

            if chat.origin == 'dataframe':
                #chat-row = 'row-reverse'
                st.dataframe(chat.message)
                

            # Check if the response is a bar chart.
            if chat.origin == 'bar':
                # chat-row = 'row-reverse'
                st.bar_chart(chat.message, height=500)
                

            # Check if the response is a line chart.
            if chat.origin == 'line':
                # chat-row = 'row-reverse'
                st.line_chart(chat.message)
                
            if chat.origin == 'AI':
                div = f"""
                    <div class="chat-row {''}">
                    <img class="chat-icon" src="{'https://ask-chatbot.netlify.app/public/ai_icon.png'}" width=32 height=32>
                    <div class="chat-bubble {'ai-bubble'}">&#8203;{chat.message}</div>
                    </div>
                    """
                st.markdown(div, unsafe_allow_html=True)

                for _ in range(3):
                    st.markdown("")   

            elif chat.origin == 'human':
                div = f"""
                    <div class="chat-row {'row-reverse'}">
                    <img class="chat-icon" src="{'https://ask-chatbot.netlify.app/public/user_icon.png'}" width=32 height=32>
                    <div class="chat-bubble {'human-bubble'}">&#8203;{chat.message}</div>
                    </div>
                    """
                st.markdown(div, unsafe_allow_html=True)

                for _ in range(3):
                    st.markdown("")

    # Create the user prompt within prompt_placeholder
    with prompt_placeholder:
        st.markdown("**Chat with me**")
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
