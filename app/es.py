import pandas as pd
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
import os
from dotenv import load_dotenv

def init_es():

    print("Initializing Elastic Search...")

    # Load environment variables
    load_dotenv()

    # Get Elastic Search URL
    ELASTIC_URL = os.getenv("ELASTIC_URL")
    # Get Elastic Search index name
    ELASTIC_INDEX_NAME = os.getenv("ELASTIC_INDEX_NAME")
    # Get Sentence Transformer model name
    SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL")
    
    # Create Elastic Search client
    es_client = Elasticsearch(ELASTIC_URL) 

    # Create index settings
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "question": {"type": "text"},
                "answer": {"type": "text"},
                "question_vector": {
                    "type": "dense_vector", 
                    "dims": 384, 
                    "index": True, 
                    "similarity": "cosine"},
                "answer_vector": {
                    "type": "dense_vector", 
                    "dims": 384, 
                    "index": True, 
                    "similarity": "cosine"},
                "question_answer_vector": {
                    "type": "dense_vector", 
                    "dims": 384, 
                    "index": True, 
                    "similarity": "cosine"},
            }
        }
    }

    # Set index name
    index_name = ELASTIC_INDEX_NAME

    # Check if index already exists
    if not es_client.indices.exists(index=index_name):

        # Delete the index if it exists and create a new one
        es_client.indices.delete(index=index_name, ignore_unavailable=True)
        es_client.indices.create(index=index_name, body=index_settings)

        # Read data from the data.csv file into a dataframe
        print("Reading data...")
        df = pd.read_csv('./data/data.csv')

        # Convert dataframe to dictionary
        records = df.to_dict(orient='records')

        # Load a model which will be used ro create the embeddings
        print("Loading model...")
        model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

        # For each record create an embedding for the question and answer field
        print("Encoding records...")
        embeddings = []
        for rec in tqdm(records):
            rec["question_vector"] = model.encode(rec["question"]).tolist()
            rec["answer_vector"] = model.encode(rec["answer"]).tolist()
            rec["question_answer_vector"] = model.encode(rec["question"] + ' ' + rec["answer"]).tolist()
            embeddings.append(rec)

        # Add all documents to index
        print("Indexing records...")
        for doc in tqdm(embeddings):
            try:
                es_client.index(index=index_name, document=doc)
            except Exception as e:
                print(e)

        # Done
        print("DONE.")
    
    else:

        print("Index is already created.")
        print("DONE.")
