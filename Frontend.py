import streamlit as st
import random
import time
from Services.prompt_generation import generate_prompt_with_problem_statement
from Services.services import build_bot, chat_with_query_engine, build_instant_rag, invoke_chain
import tempfile
import os

# Set page configuration
st.set_page_config(page_title="Custom Streamlit Application", layout="wide")

# Sidebar for API keys and configuration
st.sidebar.header("API Keys and Configuration")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
pinecone_region = st.sidebar.text_input("Pinecone Region")

# Create the api_keys dictionary
api_keys = {
    "openai_api_key": openai_api_key,
    "pinecone_api_key": pinecone_api_key,
    "pinecone_region": pinecone_region
}

# Initialize session state variables at the very beginning
if "prompt_text" not in st.session_state:
    st.session_state["prompt_text"] = ""

if "chunk_size" not in st.session_state:
    st.session_state["chunk_size"] = "300"

if "distance_metric" not in st.session_state:
    st.session_state["distance_metric"] = "Cosine"

if "reranker_topk" not in st.session_state:
    st.session_state["reranker_topk"] = "Cohere"

if "embedding_model" not in st.session_state:
    st.session_state["embedding_model"] = "text-embedding-3-small"

if "llm_model" not in st.session_state:
    st.session_state["llm_model"] = "gpt-4o"

if "language" not in st.session_state:
    st.session_state["language"] = "English"

if "settings_confirmed" not in st.session_state:
    st.session_state.settings_confirmed = False

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Problem Statement
st.header("1. Problem Statement")
problem_statement = st.text_input("Enter the problem statement:")

# 2. What do you want to build
st.header("2. What do you want to build")
build_option = st.radio(
    "Choose an option:",
    ["RAG", "Simple LLM Bot"],
    horizontal=True,
    key="build_option",
    label_visibility='collapsed'
)

# 3. Prompt
st.header("3. Prompt")
prompt_text = st.text_area("Enter your prompt:", value=st.session_state["prompt_text"])
generate_prompt = st.button("Generate Prompt")

# 4. Chunk Size
st.header("4. Chunk Size")
chunk_size_options = ["300", "800", "1200", "1400"]
chunk_size = st.radio(
    "Select chunk size:",
    chunk_size_options,
    index=0,
    horizontal=True,
    key="chunk_size",
    label_visibility='collapsed'
)

# 5. Distance Metric
st.header("5. Distance Metric")
distance_metric_options = ["Cosine", "Dot Product", "Euclidean", "Hybrid"]
distance_metric = st.radio(
    "Select distance metric:",
    distance_metric_options,
    index=0,
    horizontal=True,
    key="distance_metric",
    label_visibility='collapsed'
)

# 6. Reranker TopK
st.header("6. Reranker TopK")
reranker_options = ["Cohere", "LLM", "BM25", "SentenceTransformer"]
reranker_topk = st.radio(
    "Select reranker:",
    reranker_options,
    index=0,
    horizontal=True,
    key="reranker_topk",
    label_visibility='collapsed'
)

# 7. Embedding Model
st.header("7. Embedding Model")
embedding_model_options = ["text-embedding-3-small", "text-embedding-3-large"]
embedding_model = st.radio(
    "Select embedding model:",
    embedding_model_options,
    index=0,
    horizontal=True,
    key="embedding_model",
    label_visibility='collapsed'
)

# 8. LLM Model
st.header("8. LLM Model")
llm_model_options = ["gpt-4o", "gpt-4o-mini", "llama-3.1", "mixtral"]
llm_model = st.radio(
    "Select LLM model:",
    llm_model_options,
    index=0,
    horizontal=True,
    key="llm_model",
    label_visibility='collapsed'
)

# 9. Language
st.header("9. Language")
language_options = ["English", "Arabic", "Urdu", "Chinese"]
language = st.radio(
    "Select language:",
    language_options,
    index=0,
    horizontal=True,
    key="language",
    label_visibility='collapsed'
)

# 10. Document Upload
st.header("10. Upload Your Document")
uploaded_file = st.file_uploader(
    label="Choose a PDF, TXT, DOCX, or CSV",
    type=["pdf", "txt", "docx", "csv"]
)

if uploaded_file is not None:
    # Show a message with the filename
    st.success(f"File '{uploaded_file.name}' uploaded!")
    # You can also display basic info, e.g. size in KB:
    file_size_kb = uploaded_file.size / 1024
    st.write(f"Size: {file_size_kb:.1f} KB")
else:
    st.info("No file uploaded yet.")

confirm_settings = st.button("Confirm Settings")

# Create the bot_config dictionary
bot_config = {
    "problem_statement": problem_statement,
    "build_option": build_option,
    "prompt_text": prompt_text,
    "chunk_size": int(chunk_size),
    "distance_metric": distance_metric,
    "reranker_topk": reranker_topk,
    "embedding_model": embedding_model,
    "llm_model": llm_model,
    "language": language,
    "top_k": 5  # Default value, can be adjusted later
}

# 10. Chat Bot Box
st.header("10. Chat Bot")

# -----------------------------------------------------------------------------
# Create a placeholder and a function to render the chat in one place.
chat_placeholder = st.empty()

# def render_chat():
#     """
#     Clear the placeholder, then re-render the entire conversation.
#     """
chat_placeholder.empty()
with chat_placeholder.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
# -----------------------------------------------------------------------------

# If the user clicks "Generate Prompt", call your prompt-generation function.
if generate_prompt:
    api_key = api_keys["openai_api_key"]
    problem_stmt = bot_config['problem_statement']
    bot_type = bot_config["build_option"]
    # Generate new prompt
    result = generate_prompt_with_problem_statement(api_key, problem_stmt, bot_type)

    # Update session_state and re-run
    st.session_state["prompt_text"] = result
    bot_config["prompt_text"] = result
    st.rerun()

# Once settings are confirmed, build the bot and store in session
if confirm_settings:
    if uploaded_file is None:
        st.error("Please upload a document before confirming settings.")
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_path = os.path.join(tmpdirname, uploaded_file.name)
            # Write the uploaded bytes to a file
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            chain, retriever = build_instant_rag(bot_config, api_keys, temp_path)
            st.session_state.chain = chain
            st.session_state.retriever = retriever
            st.session_state.settings_confirmed = True
            st.success("Settings confirmed. You can now start chatting.")

# Accept user input
user_input = st.chat_input("Type your message here...")

if user_input:
    # If settings are not confirmed, warn the user
    if not st.session_state.settings_confirmed:
        assistant_message = "Please confirm settings before chatting."
        st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Generate response from the bot
        chain = st.session_state.get("chain", None)
        retriever = st.session_state.get("retriever", None)
        if chain and retriever:
            response = invoke_chain(chain, retriever, bot_config, user_input, api_keys)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            # If for some reason the bot wasn't built
            response = "Bot not built yet. Please confirm settings."
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Re-render chat with all messages
    # render_chat()
# else:
#     # If no new user input, just render the existing chat as-is
#     render_chat()
chat_placeholder.empty()
with chat_placeholder.container():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])