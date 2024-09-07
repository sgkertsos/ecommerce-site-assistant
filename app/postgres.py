import psycopg2
import os
from dotenv import load_dotenv


# This function initializes a connection
def init_connection():

    # Load environment variables
    load_dotenv()

    # Get postgres environment variables
    POSTGRES_HOST=os.getenv("POSTGRES_HOST")
    POSTGRES_DB=os.getenv("POSTGRES_DB")
    POSTGRES_USER=os.getenv("POSTGRES_USER")
    POSTGRES_PASSWORD=os.getenv("POSTGRES_PASSWORD")
    POSTGRES_PORT=os.getenv("POSTGRES_PORT")

    # Connect to postgres
    conn = psycopg2.connect(database = POSTGRES_DB, 
                        user = POSTGRES_USER, 
                        host= POSTGRES_HOST,
                        password = POSTGRES_PASSWORD,
                        port = POSTGRES_PORT)

    # Return connection
    return conn 

# This function creates the tables if they don't exist
def init_postgres():

    print("Initializing Postgres...")

    # Initialize connection
    conn = init_connection()
    
    # Open a cursor 
    cursor = conn.cursor()
      
    # Create dialogs table
    cursor.execute("""CREATE TABLE IF NOT EXISTS dialogs(
                id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                response_time FLOAT NOT NULL,
                prompt_tokens INT NOT NULL,
                completion_tokens INT NOT NULL,
                total_tokens INT NOT NULL, 
                eval_prompt_tokens INT NOT NULL,
                eval_completion_tokens INT NOT NULL,
                eval_total_tokens INT NOT NULL,
                relevance TEXT NOT NULL,
                total_cost FLOAT NOT NULL,
                eval_total_cost FLOAT NOT NULL,
                tstz TIMESTAMPTZ NOT NULL);
                """)
    
    # Create feedback table
    cursor.execute("""CREATE TABLE IF NOT EXISTS feedback(
                id SERIAL PRIMARY KEY,
                dialog_id TEXT REFERENCES dialogs(id),
                feedback INT NOT NULL,
                tstz TIMESTAMPTZ NOT NULL);
                """)
    
    # Commit changes
    conn.commit()

    # Close cursor and database connection
    cursor.close()
    conn.close()

    print("DONE.")

# This function inserts a feedback
def insert_feedback(dialog_id, feedback):
    
    # Initialize connection
    conn = init_connection()

    # Open a cursor 
    cursor = conn.cursor()

    # Insert record
    sql = f"insert into feedback (dialog_id, feedback, tstz) values (%s, %s, NOW())"
    cursor.execute(sql, (dialog_id, feedback))

    # Commit record
    conn.commit()

    # Close cursor and database connection
    cursor.close()
    conn.close()

# This function inserts a chat
def insert_dialog(id, question, answer):

    # Initialize connection
    conn = init_connection()

    # Open a cursor 
    cursor = conn.cursor()

    # Insert record
    sql = f"insert into dialogs (id, question, answer, response_time, prompt_tokens, completion_tokens, total_tokens, eval_prompt_tokens, eval_completion_tokens, eval_total_tokens, relevance, total_cost, eval_total_cost, tstz) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())"

    # Execute cursor
    cursor.execute(sql, (id, question, answer["answer"], answer["response_time"], answer["prompt_tokens"], answer["completion_tokens"],  answer["total_tokens"], answer["eval_prompt_tokens"], answer["eval_completion_tokens"],  answer["eval_total_tokens"], answer["relevance"], answer["total_cost"], answer["eval_total_cost"]))

    # Commit record
    conn.commit()

    # Close cursor and database connection
    cursor.close()
    conn.close()