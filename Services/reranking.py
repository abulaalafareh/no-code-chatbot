import cohere
from sentence_transformers import CrossEncoder
import openai
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
# from text2vec import CrossEncoder as Text2VecCrossEncoder


def rerank_crossencoder_minilm(documents, query, top_k=5):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [(query, doc) for doc in documents]
    scores = model.predict(pairs)

    scored_docs = sorted(zip(scores, documents), reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

def rerank_bge_base(documents, query, top_k=5):
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")

    scored_docs = []
    for doc in documents:
        inputs = tokenizer(f"{query} [SEP] {doc}", return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
            score = logits[0].item()
        scored_docs.append((score, doc))

    scored_docs.sort(reverse=True)
    return [doc for _, doc in scored_docs[:top_k]]

# def rerank_text2vec(documents, query, top_k=5):
#     model = Text2VecCrossEncoder("shibing624/text2vec-base-multilingual")
#     pairs = [(query, doc) for doc in documents]
#     scores = model.predict(pairs)

#     scored_docs = sorted(zip(scores, documents), reverse=True)
#     return [doc for _, doc in scored_docs[:top_k]]

def rerank_cohere(documents, query, top_k=5, api_key=None):
    if api_key is None:
        raise ValueError("Cohere API key must be provided.")
    documents=[{"text": doc} for doc in documents]
    print("documents", documents)
    co = cohere.Client(api_key)
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[{"text": doc["page_content"]} for doc in documents],
        top_n=top_k
    )
    return [res.document["text"] for res in response.results]

def llm_rerank(api_key, documents, query, top_k=5):
    system_prompt = """You are given a query and a document. Your job is to determine whether the query can be answered using the document.
        Give the document a score from 1 to 10 based on how relevant it is to answering the query.

        Return your response in JSON format like this:
        {
            "score": <your score>
        }
        """
    openai.api_key = api_key
    scored_documents = []

    for document in documents:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-3.5-turbo, adjust based on your use
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"question: {query}\ndocument: {document}"}
            ]
        )

        try:
            json_response = json.loads(response.choices[0].message.content)
            score = float(json_response["score"])
        except (json.JSONDecodeError, KeyError, ValueError):
            score = 0  # fallback score if parsing fails

        scored_documents.append((score, document))

    # Sort by score descending and return top_k documents
    scored_documents.sort(reverse=True, key=lambda x: x[0])
    return [doc for _, doc in scored_documents[:top_k]]

