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
from uuid import uuid4
from Services.vectorstore import Vectorstore
from Services.reranking import llm_rerank

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

def build_instant_rag(bot_config, api_keys, file_path=None):

    template = """Your Role : {role}, 
    
    User's query {query},
    Context to answer the question from {summaries}

    Please answer the user question in {language} language.
     
    If the query cannot be answered from the given context just say "I don't Know" """

    print(bot_config)
    # llm model
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key = api_keys['openai_api_key'])
    # load documents, chunk documents
    docs = Load_and_Chunk.doc_and_docx_loader(file_path=file_path, chunk_size=bot_config['chunk_size'], chunk_overlap=50)
    # embed documents
    vector_store = Vectorstore(embedding_model=bot_config['embedding_model'])    
    faiss = vector_store.initialize_faiss(distance_metric="EUCLIDEAN_DISTANCE")
    
    # create ids
    uuids = [str(uuid4()) for _ in range(len(docs))]

    faiss.add_documents(documents=docs, ids=uuids)

    search_kwargs={'k': bot_config['top_k']}
    # Retrieve and generate using the relevant snippets of the blog.
    retriever = faiss.as_retriever(search_kwargs = search_kwargs)

    prompt = PromptTemplate(template=template, input_variables=["role","summaries","query"])

    rag_chain = prompt | llm

    return rag_chain, retriever

def invoke_chain(chain, retriever, bot_config, query, api_keys=None):

    retrieved_docs = retriever.invoke(query)
    prompt = bot_config['prompt_text']
    # retrieved_docs = rerank_cohere(retrieved_docs, query, bot_config['top_k'], "19A6NzN0dyartCsq9hkGgth0x1cy7PKdoxes6Dwv")
    # retrieved_docs = llm_rerank(api_keys["openai_api_key"],retrieved_docs, query, bot_config['top_k'])
    summaries =  "\n\n----------------------------\n\n".join(doc.page_content for doc in retrieved_docs)
    print("Summaries are", summaries)
    response = chain.invoke({"role":prompt, "summaries":summaries, "query":query, "language":bot_config['language']})
    print("Response is", response)
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
