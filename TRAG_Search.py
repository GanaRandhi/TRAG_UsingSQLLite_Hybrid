import sqlite3
import os
import streamlit as st
import logging
import google.generativeai as genai
import pandas as pd

from cmd import PROMPT
from datetime import timedelta, datetime
from random import randint
from random import choice as rc
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import SQLiteVec
from tqdm.auto import tqdm
from langchain_huggingface import HuggingFaceEmbeddings


# This function will return a random datetime between two datetime objects.
def random_date(start, end):
    return start + timedelta(seconds=randint(0, int((end - start).total_seconds())))


class LoggingFormatter(logging.Formatter):
    """
    Custom logging formatter to add colors and styles to log messages."""
    # Colors
    black = "\x1b[30m"
    red = "\x1b[31m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"
    gray = "\x1b[38m"
    # Styles
    reset = "\x1b[0m"
    bold = "\x1b[1m"

    COLORS = {
        logging.DEBUG: gray + bold,
        logging.INFO: blue + bold,
        logging.WARNING: yellow + bold,
        logging.ERROR: red,
        logging.CRITICAL: red + bold
    }
    
    def format(self, record):
        log_color = self.COLORS[record.levelno]
        format = "(black){asctime}(reset) (levelcolor){levelname:<8}(reset) (green){name}(reset) {message}"
        format = format.replace("(black)", self.black + self.bold)
        format = format.replace("(reset)", self.reset)
        format = format.replace("(levelcolor)", log_color)
        format = format.replace("(green)", self.green + self.bold)
        formatter = logging.Formatter(format, "%Y-%m-%d %H:%M:%S", style="{")
        return formatter.format(record)

database_file = "./dist/northwind.db" #@param

def get_db_connection():
           
    with sqlite3.connect(database_file) as conn:
        conn.row_factory = sqlite3.Row
        return conn

def create_dbs():
    with open("./sql/create.sql", 'r') as sql_file:
        sql_script = sql_file.read()
    cursor.executescript(sql_script)
    st.write("Database created and script executed successfully.")    
    

def get_all_tables_data():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]   
    
    all_tables_data = {}
    schema_info = {}
    try:
        for table_name in table_names:
            # Read each table into a Pandas DataFrame
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            all_tables_data[table_name] = df
            
            cursor.execute(f'PRAGMA table_info("{table_name}");')
            schema_info  = cursor.fetchone()
    except sqlite3.Error as e:
        st.write(f"An error occurred: {e}")
        
    cursor.close()
    conn.close()
    return table_names, all_tables_data, schema_info 

# Connect to the DB
try:        
    conn = get_db_connection()
    cursor = conn.cursor()
    if os.path.isfile(database_file):
        print(f"The database exists.")
    else:
        create_dbs()  
    
    cursor.close()
    conn.close()
except sqlite3.Error as e:
    st.write(f"An error occurred: {e}")

all_tables, all_tables_data, database_schema = get_all_tables_data()

# if all_tables:
#     st.subheader("Available Tables Names:")
#     for table_name in all_tables[1:]:
#         st.write(f"**{table_name}**")
# else:
#     st.info("No tables found in the database.")   

# if all_tables_data:
#     st.subheader("Available Tables Data:")
#     for table_name, df in all_tables_data.items():
#         st.write(f"**{table_name}** :- Total rows:  {len(df)}")
#         st.dataframe(df) # Display each DataFrame in an interactive table
# else:
#     st.info("No tables found in the database.")

# if database_schema:
#     st.subheader("Creating DB Schema")
#     # st.write(database_schema)
#     for schema in database_schema:
#         st.write(schema)

load_dotenv()
# Configure Google Generative AI
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    print("Google Generative AI configured successfully.")
except Exception as e:
    print(f"ERROR: Failed to configure Google Generative AI: {e}")
    print("Please ensure GOOGLE_API_KEY is set in your .env file or environment.")
    # You might want to exit or disable LLM functionality if API key is missing
    exit() # Exiting for critical dependency
    

# Use a suitable Gemini model
LLM_MODEL_NAME = 'gemini-2.5-flash' # Or 'gemini-pro', 'gemini-2.5-pro' depending on needs
# Initialize the GenerativeModel
model = genai.GenerativeModel(LLM_MODEL_NAME)

engine = create_engine(f'sqlite:///{database_file}')

CLEANING_PATTERN = r'[^a-zA-Z0-9]'

prompt = '''
Your persona:
You are an AI assistant that can answer questions about a movie database.
You are a helpful assistant that helps user to find information from northwind database.
The database contains information about Categories, CustomerCustomerDemo, CustomerDemographics, Customers, Employees, EmployeeTerritories, Order Details, Orders, Products, Regions, 
Shippers, Suppliers, and Territories.
Your task is to answer the user's query based on the provided database information.
Your persona is polite, friendly and helpful.

While Trying to solve the customer's query, you can use the following information:
 - You can ask clarifying questions to understand the user's needs better.
 - You can use the northwind database that match the user's request.
 - You can provide additional information about the information, such as famous orders, products, and suppliers.
 - You can suggest alternatives or modifications to the information based on the user's preferences.

User Query: {user_query}

Generate a SQL query to retrieve the requested information from the database.
database: sqlite:///northwind.db
schema details:
{database_schema}
'''

special_instructions = '''

SPECIAL INSTRUCTIONS:

MAKE AN ERROR IN THE SQL FORMATION, I WANT TO TEST ERROR HANDLING THE SQL QUERY
'''

error_handling_prompt = '''
You are a database response handler for business users.
Under stand the user query, the generated sql and the error that has occured.

if you can fix the query provide the fixed query and query alone. add in a comment: `--fixed_query` at the end to mark fix

else provide a polite way to say that there was an error in the query and apologise.
your response may be seen by LLMs further down the line.

User Query: {user_query}

SQL Query: {sql}

schema: {database_schema}

Error: {error}
'''

user_queries =[
    #f"Who are top 5 customers ?",
    #f"Get me the details of the two oldest employees along with their age.",
]
sql_queries = []

for user_query in user_queries:
    resp = model.generate_content(
        prompt.format(user_query=user_query, database_schema=database_schema) + special_instructions
        ).text
    sql = resp.strip().split('```sql')[1].split(
        '```')[0].strip()  # Extract SQL query from response
    sql_queries.append(sql)
    st.markdown("---")
    st.write(user_query)
    print(sql, '\n\n')
    db_response = None
    try:
        
        db_response = pd.read_sql(sql, engine).to_markdown(index=False)
        #st.write(db_response)
        #st.markdown("---")
    except Exception as e:
        db_response = model.generate_content(
            error_handling_prompt.format(
                error=str(e),
                user_query=user_query,
                sql=sql,
                database_schema=database_schema
              )
          ).text
        #st.write(db_response)
        db_response = db_response.strip().split('```sql')[1].split(
        '```')[0].strip()
        st.write(pd.read_sql(db_response, engine))
        st.markdown("---")
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

stores = {}
for table_name in tqdm(all_tables):
    data = pd.read_sql(
        f'''
        SELECT * FROM "{table_name}" LIMIT 3
        ''',
        engine
    )
    # st.subheader(table_name)
    # print(data.head().to_markdown())
    # st.write(data.head().to_markdown())

    data = [str(i) for i in data.to_dict('records')]

    connection = SQLiteVec.create_connection(db_file=f"./vectorDB/{table_name}_vec.db")

    vector_store = SQLiteVec(
        table="intelligence",
        embedding=embedding_function,
        connection=connection,
        )

    _ = vector_store.add_texts(texts=data)
    stores[table_name] = vector_store
    
vector_store = stores['Categories']
query = 'Pictures'

for doc in vector_store.similarity_search_with_score(query,):
    st.write(doc)
    st.markdown("---")