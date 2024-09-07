# Import the necessary libraries
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from openai import OpenAI
import os
from dotenv import load_dotenv
import time
import json

# Load environment variables
load_dotenv()

# Load a model for encoding embeddings
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL")
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

# Elastic Search
ELASTIC_URL = os.getenv("ELASTIC_URL")
es_client = Elasticsearch(ELASTIC_URL) 
ELASTIC_INDEX_NAME = os.getenv("ELASTIC_INDEX_NAME")
index_name = ELASTIC_INDEX_NAME

# Open AI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Open AI model
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# Create Open AI client
ai_client = OpenAI()

# Function to calculate llm cost
def calculate_cost(prompt_tokens, completion_tokens):
    
    OPENAI_PROMPT_COST = os.getenv("OPENAI_PROMPT_COST")
    OPENAI_COMPLETION_COST = os.getenv("OPENAI_COMPLETION_COST")

    prompt_cost = (prompt_tokens / 1000) * float(OPENAI_PROMPT_COST)
    completion_cost = (completion_tokens / 1000) * float(OPENAI_COMPLETION_COST)
    total_cost = prompt_cost + completion_cost
    
    return total_cost

# Define text search function
def elastic_text_search(query):
    
    # Create query
    search_query = {
        "size": 2,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["id", "question", "answer"],
                        "type": "best_fields"
                    }
                }
            }
        }
    }


    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs

# Define vector search function
def elastic_vector_search(field, query):

    # Encode query to a vector
    query_v = model.encode(query)

    # Construct search query
    search_query = {
        "field": field,
        "query_vector": query_v,
        "k": 2,
        "num_candidates": 10000, 
    }

    response = es_client.search(index=index_name, knn=search_query, source=["id", "question", "answer"])
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        result_docs.append(hit['_source'])
    
    return result_docs
    
# Define build prompt function
def build_prompt(query, search_results):
    prompt_template = """
    You're a eCommerce site chatbot. Answer the QUESTION based on the CONTEXT from our database.
    Use only the facts from the CONTEXT when answering the QUESTION. If you don't find an answer
    just say that you are sorry but you don't have an answer to this question.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    context = ""

    for doc in search_results:
        context = context + f"question: {doc['question']}\nanswer: {doc['answer']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt
    
# Define llm function
def llm(prompt, model=OPENAI_MODEL):

    # Get start time
    start_time = time.time()

    # Get response from LLM
    response = ai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    # Get end time
    end_time = time.time()

    # Calculate response time
    response_time = end_time - start_time

    # LLM answer    
    answer = response.choices[0].message.content

    # LLM tokens
    tokens = {
        'prompt_tokens': response.usage.prompt_tokens,
        'completion_tokens': response.usage.completion_tokens,
        'total_tokens': response.usage.total_tokens
    }

    # LLM cost
    cost = calculate_cost(tokens['prompt_tokens'], tokens['completion_tokens'])

    return answer, tokens, response_time, cost

# Define relevance function
def get_relevance(question, answer):
    prompt_template = """
    You evaluate our RAG system. One of your tasks is to analyze if there is a relevance
    between the given question and the generated answer. You have to classify the relevance
    as "Relevant", "Non relevant" or "Partly relevant"

    Data for evaluation:

    Question: {question}
    Generated Answer: {answer}

    Provide your evaluation in parsable JSON without using code blocks:

    {{
      "Relevance": "Relevant" | "Non relevant" | "Partly relevant"
    }}
    """.strip()

    # Create prompt
    prompt = prompt_template.format(question=question, answer=answer)

    # Get values
    eval, tokens, _, cost = llm(prompt, model=OPENAI_MODEL)
   
    # Return values
    # Check if the answer can be parsed
    try:
        print(answer)
        evaluate_json = json.loads(eval)
        return evaluate_json["Relevance"], tokens, cost
    except json.JSONDecodeError:
        return "UNKNOWN", tokens
    
# Define rag function
def rag(query, model=OPENAI_MODEL) -> str:
    
    # Get results from elastic database
    search_results = elastic_vector_search("question_vector", query)
    
    # Build a prompt
    prompt = build_prompt(query, search_results)
    
    # Get answer from LLM
    answer, tokens, response_time, cost = llm(prompt, model=model)

    # Get relevance from LLM
    relevance, eval_tokens, eval_total_cost = get_relevance(query, answer)
    
    return {
        'answer': answer,
        'response_time': response_time,
        'relevance': relevance,
        'prompt_tokens': tokens["prompt_tokens"],
        'completion_tokens': tokens['completion_tokens'],
        'total_tokens': tokens['total_tokens'],
        'eval_prompt_tokens': eval_tokens['prompt_tokens'],
        'eval_completion_tokens': eval_tokens['completion_tokens'],
        'eval_total_tokens': eval_tokens['total_tokens'],
        'total_cost': cost,
        'eval_total_cost': eval_total_cost
    }
