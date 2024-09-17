import openai
from langchain_community.embeddings import OpenAIEmbeddings


class Embeddings:

    def __init__(self,openai_api_key, embedding_model = "text-embedding-3-small"):
        openai.api_key = openai_api_key
        self.embedding_model = embedding_model
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model=embedding_model)

    def embed_query(self, query):

        response = openai.embeddings.create(
            input=query,
            model=self.embedding_model
        )

        return(response.data[0].embedding)
    
    def embed_documents(self, documents):

        embeddings = self.embeddings.embed_documents(texts=documents)

        return embeddings