import cohere

def cohere_rerank(cohere_api_key, query, retrieved_documents, top_k):
    # Initialize the Cohere client
    client = cohere.Client(api_key=cohere_api_key)


    # Convert the retrieved nodes to a list of responses
    documents = [doc.metadata['text'] for doc in retrieved_documents]

    # Rerank the responses
    reranked_responses = client.rerank(
    model="rerank-english-v3.0",
    query=query,
    documents=documents,
    top_n=top_k
    )
    return reranked_responses

