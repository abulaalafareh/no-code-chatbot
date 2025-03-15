from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain.vectorstores import FAISS



EMBEDDING_SIZE = {
    'text-embedding-3-small': 1536,
    'text-embedding-3-large': 3072,
    'text-embedding-ada-002': 1536,
}

class Vectorstore:

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.embed = OpenAIEmbeddings(model=embedding_model)
        self.embedding_size = EMBEDDING_SIZE[embedding_model]

    def initialize_faiss(self,distance_metric):
        index = faiss.IndexFlatL2(self.embedding_size)
        vector_store = FAISS(
            embedding_function=self.embed,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
            distance_strategy= distance_metric
        )
        return vector_store 