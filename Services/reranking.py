import cohere
from llama_index.core.postprocessor import SentenceTransformerRerank
import openai
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

def sentence_transformer_rerank(documents, query, top_k=5):
    rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=top_k
    )
    reranked_responses = rerank.postprocess_nodes(nodes=documents,query_str=query)
    return reranked_responses

def llm_rerank(api_key, documents, query, top_k=5):
    system_prompt = """ You are given a query and a document you job is to see wether the query can be answered using the document
    you have to give document a score from 1 to 10. 
    Give score based on how relevant the document is.

    return your response in JSON format:
    {{
        score : <your score>
    }}
    
    """
    openai.api_key = api_key
    scores = []
    scores_dict = {}
    for document in documents:
        response = openai.chat.completions.create(
                    response_format={"type":"json_object"},
                    messages = [
                        {"role":"assistant", "content":system_prompt},
                        {"role":"user", "content":f""" question:{query}, document:{document}"""}
                        ]
                )
        json_response = response.choices[0].message.content
        score = json_response['score']
        scores_dict[score] = document
        scores.append(score)
    
    scores = scores.sort()
    new_documents = []
    for score in scores:
        new_documents.append(scores_dict[score])
    return new_documents[:top_k]
