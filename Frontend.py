import streamlit as st
import random
import time

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
prompt_text = st.text_area("Enter your prompt:")
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
embedding_model_options = ["Test-embedding-3-small", "Test-embedding-3-large"]
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

# Create the bot_config dictionary
bot_config = {
    "problem_statement": problem_statement,
    "build_option": build_option,
    "prompt_text": prompt_text,
    "chunk_size": chunk_size,
    "distance_metric": distance_metric,
    "reranker_topk": reranker_topk,
    "embedding_model": embedding_model,
    "llm_model": llm_model,
    "language": language
}

# 10. Chat Bot Box
st.header("10. Chat Bot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to simulate streaming response
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi! Is there anything I can help you with?",
            "Do you need assistance with something?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.1)  # Adjust the sleep time for faster or slower typing effect

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for chunk in response_generator():
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    print(st.session_state)
    # print(bot_config)
    # print(api_keys)