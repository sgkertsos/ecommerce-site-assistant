import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv
from tqdm.auto import tqdm
import json
from rag import rag, elastic_text_search, elastic_vector_search
from sentence_transformers import SentenceTransformer

# Load environment variables
load_dotenv()

# Get Sentence Transformer model name
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL")
model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

# Open AI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Open AI model
OPENAI_MODEL = os.getenv("OPENAI_MODEL")

# Create Open AI client
ai_client = OpenAI()

# This function is used to generate the ground truth data
def generate_ground_truth_data():

    # Read data from the csv file
    df = pd.read_csv('./data/data.csv')

    # Populate documents
    records = df.to_dict(orient='records')

    # Dictionary to store the results
    results = {}

    # Iterate through all records
    for rec in tqdm(records): 
        rec_id = rec['id']
        if rec_id in results:
            continue
        
        # Generate 5 questions for each record
        questions = generate_questions(rec)
        results[rec_id] = questions    
   
    # Parse the results
    parsed_results = {}

    for rec_id, json_questions in results.items():
        parsed_results[rec_id] = json.loads(json_questions)

    # Create a record index
    rec_index = {r['id']: r for r in records}

    # Create a dataframe
    final_results = []

    for rec_id, questions in parsed_results.items():
        for q in questions:
            final_results.append((q, rec_id))

    df = pd.DataFrame(final_results, columns=['question', 'id'])

    # Save dataframe to a csv file
    df.to_csv('./data/ground-truth-data.csv', index=False)

# This function is used to generate questions for a specific answer
def generate_questions(doc):

    # This is the template to generate the prompt
    prompt_template = """
    Create 5 questions a user can ask based on a FAQ record. The record should contain the answer to the questions.

    The record:

    question: {question}
    answer: {answer}

    Provide the output in parsable JSON without using code blocks:

    ["question1", "question2", ..., "question5"]
    """.strip()

    # Generate prompt
    prompt = prompt_template.format(**doc)

    response = ai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    json_response = response.choices[0].message.content
    return json_response

# This function loads the ground truth data
def load_ground_truth_data():

    # Load csv data into a dataframe
    gtd_df = pd.read_csv("./data/ground-truth-data.csv")

    # Convert dataframe to dictionary
    gtd_dict = gtd_df.to_dict(orient='records')

    # Return dictionary
    return gtd_dict

# This function evaluates our text search retrieval
def evaluate_text_search_retrieval():
    
    # Load ground truth data
    gtd = load_ground_truth_data()

    # List to store all checks
    total_relevance = []

    # Iterate through records
    for rec in tqdm(gtd):
        # Get record id
        rec_id = rec['id']
        # Perform a text search with the record question
        results = elastic_text_search(query=rec['question'])
         # List to store if record is relevant
        relevance = []
        for r in results:
            relevance.append(r['id'] == rec_id)

        # Add each record relevance to total relevance
        total_relevance.append(relevance)

    # Show hit rate
    hit_rate = calculate_hit_rate(total_relevance=total_relevance)
    mrr = calculate_mrr(total_relevance=total_relevance)

    return hit_rate, mrr

# This function evaluates our vector search retrieval
def evaluate_vector_search_retrieval(field):

    # Load ground truth data
    gtd = load_ground_truth_data()

    # List to store all checks
    total_relevance = []

    # Iterate through records
    for rec in tqdm(gtd):
        # Get record id
        rec_id = rec['id']
        # Perform a text search with the record question
        results = elastic_vector_search(field=field, query=rec['question'])
         # List to store if record is relevant
        relevance = []
        for r in results:
            relevance.append(r['id'] == rec_id)

        # Add each record relevance to total relevance
        total_relevance.append(relevance)

    # Show hit rate
    hit_rate = calculate_hit_rate(total_relevance=total_relevance)
    mrr = calculate_mrr(total_relevance=total_relevance)

    return hit_rate, mrr

# Function to calculate Hit Rate.
# If id is in results increase hit by 1
# Finally divide total hits by length of total relevance
def calculate_hit_rate(total_relevance):
    hits=0
    for row in total_relevance:
        for col in row:
            if col == True:
                hits +=1
    return hits / len(total_relevance)

# Function to calculate MRR
# Works almost as Hit Rate but instead of increasing hits by 1 
# the increase rate takes the position in consideration.
# Rate is 1 / Position
def calculate_mrr(total_relevance):
    hits=0.0
    for row in total_relevance:
        for col in row:
            if row[col] == True:
                hits = hits + 1 / (col + 1) 
    return hits / len(total_relevance)

# This function is used to generate the offline RAG evaluation data
def generate_offline_rag_evaluation_data(model):
    
    # Read data from the data.csv file into a dataframe
    df = pd.read_csv('./data/data.csv')

    # Convert dataframe to dictionary
    records = df.to_dict(orient='records')

    # Load Ground Truth data
    ground_truth = load_ground_truth_data()

    # For each question in ground truth data generate an answer
    answers = []

    for record in ground_truth:
        llm_answer = rag(record['question'], model=model)
        record_id = record['id']
        original_answer = records[record_id]['answer']
        question = record['question']
        answers.append({
            'id': record_id,
            'question': question,
            'llm_answer': llm_answer,
            'original_answer': original_answer 
        })

        # Create results data frame
        df = pd.DataFrame([{'id': record['id'], 
                            'question': record['question'], 
                            'llm_answer': record['llm_answer']['answer'], 
                            'original_answer': record['original_answer']} for record in answers])
        
        # Save results to csv file
        # gpt-35-turbo
        if model == 'gpt-3.5-turbo':
            df.to_csv('./data/gpt-35-turbo-results.csv')

        # gpt-4o-mini
        if model == 'gpt-4o-mini':
             df.to_csv('./data/gpt-4o-mini-results.csv')

# This function calculates cosine similarity
def calculate_cosine_similarity(record):
    
    # Get original and llm answer
    original_answer = record['original_answer']
    llm_answer = record['llm_answer']

    # Calculate vectors
    original_answer_v = model.encode(original_answer)
    llm_answer_v = model.encode(llm_answer)

    # Return similarity
    return llm_answer_v.dot(original_answer_v)

# This function calculates similarities
def calculate_similarities(model):
    # Read results
    # gpt-35-turbo
    if model == 'gpt-3.5-turbo':
        df = pd.read_csv('./data/gpt-35-turbo-results.csv')
    
    # gpt-4o-mini
    if model == 'gpt-4o-mini':
        df = pd.read_csv('./data/gpt-4o-mini-results.csv')
    
    # Convert dataset to dict
    results = df.to_dict(orient='records')

    # Calculate similarity for each record
    similarities = []

    for record in results:
        similarity = calculate_cosine_similarity(record)
        similarities.append(similarity)

    # Add cosine column to Dataframe
    df['cosine'] = similarities

    # Describe column
    column_description = dict(df['cosine'].describe())

    # Return column column description
    return column_description, df['cosine']