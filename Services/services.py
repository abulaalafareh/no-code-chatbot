from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, get_response_synthesizer
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
from langchain_core.prompts import PromptTemplate
from llama_index.llms.openai import OpenAI

from Services.document_processing import Load_and_Chunk
# Function to simulate streaming response
def build_bot(bot_config, api_keys):
    print("in basic bot building")
    print("bot_config", bot_config)
    llm = OpenAI(model="gpt-4")
    if bot_config['build_option'] == "RAG":
        # Settings.chunk_size = bot_config['chunk_size']
        # Settings.embed_model = bot_config['embedding_model']
        # Settings.llm = bot_config['llm_model']
        data = SimpleDirectoryReader(r"C:\Users\ASUS\LLM_Project\data")
        documents = data.load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(llm=llm)
        # query_engine.query()
        return query_engine

def chat_with_query_engine(query_engine,query):
    response = query_engine.query(query)
    return response  

def build_instant_rag(bot_config, api_keys):

    template = """Your Role : {role}, 
    
    User's query {query},
    Context to answer the question from {summaries}
     
    If the query cannot be answered from the given context just say "I don't Know" """

    file_path = r"C:\Users\ASUS\LLM_Project\data\metagpt.pdf"

    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key = api_keys['openai_api_key'])
    loader = Load_and_Chunk.pdf_loader(file_path=file_path, chunk_size=bot_config['chunk_size'], chunk_overlap=50)

    docs = loader
    vector_store = Chroma()
    vectorstore = vector_store.from_documents(documents=docs, embedding=OpenAIEmbeddings(model=bot_config['model']))

    search_kwargs={'k': bot_config['top_k']}
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = vectorstore.as_retriever(search_kwargs = search_kwargs)
    # retrieved_docs = retriever.invoke(query)

    # prompt = bot_config['prompt']
    
    # summaries =  "\n\n----------------------------\n\n".join(doc.page_content for doc in retrieved_docs)

    # print(summaries)

    prompt = PromptTemplate(template=template, input_variables=["role","summaries","query"])

    rag_chain = prompt | llm

    return rag_chain, retriever

def invoke_chain(chain, retriever, bot_config, query):

    retrieved_docs = retriever.get_relevant_documents(query)

    prompt = bot_config['prompt']
    
    summaries =  "\n\n----------------------------\n\n".join(doc.page_content for doc in retrieved_docs)

    response = chain.invoke({"role":prompt, "summaries":summaries, "query":query})
    return response.content

# bot_config = {
#     "top_k":5,
#     "chunk_size":300,
#     "model":"text-embedding-3-small",
#     'metric':'euclidean',
#     "prompt":"Answer my questions"
# }
# api_keys = {
#     "openai_api_key":os.getenv("OPENAI_API_KEY")
# }
# query = "what is this document about"
# chain, retriever = build_instant_rag(bot_config,api_keys)
# response = invoke_chain(chain, retriever, bot_config, query)
# print("Result is",response)
