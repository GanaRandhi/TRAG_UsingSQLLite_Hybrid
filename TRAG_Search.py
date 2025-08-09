import sqlite3
import os
from typing import Any
import streamlit as st
import logging
import google.generativeai as genai
import pandas as pd
from typing import Optional
# import json
# from io import BytesIO
# import base64

from asyncio.windows_events import NULL
from cmd import PROMPT
from datetime import timedelta, datetime
from random import randint
from random import choice as rc
from dotenv import load_dotenv
from sqlalchemy import create_engine
#from langchain_community.embeddings import OllamaEmbeddings
# from langchain_community.vectorstores import SQLiteVec
# from tqdm.auto import tqdm
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

options = os.listdir("./dist/")
selected_option = st.selectbox("Select an DB:", options) 
st.write(selected_option)
database_file = "./dist/" + str(selected_option) #@param
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
    try:
        for table_name in table_names:
            # Read each table into a Pandas DataFrame
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            all_tables_data[table_name] = df
    except sqlite3.Error as e:
        st.write(f"An error occurred: {e}")
        
    cursor.close()
    conn.close()
    return table_names, all_tables_data

def get_schema(table_name, formatted_string, schema):
    """
    Retrieves the schema of a specified table in the SQLite database.

    Args:
        table_name (str): The name of the table to retrieve schema for.
        engine: SQLAlchemy engine connected to the SQLite database.

    Returns:
        str: Formatted string representing the table schema.
    """
    conn = get_db_connection()
    database_schema.setdefault(table_name, [])
    columns_info = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    table_columns = [(col[1], col[2]) for col in columns_info]    
    formatted_string += f"{table_name}: \n"
    for col_name, col_type in table_columns:
        formatted_string += f" {col_name} ({col_type})"
        database_schema[table_name].append(f"{col_name} ({col_type})")
    return formatted_string, database_schema


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

all_tables, all_tables_data = get_all_tables_data()
db_schema = ""
database_schema={}
if all_tables:
    for table_name in all_tables:
        db_schema, database_schema = get_schema(table_name, db_schema, database_schema)
    #st.write(f"DB Schema in text: {db_schema}")

###  All Table Names, Tabular Data, Database Schema
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
#     #st.write(database_schema)
#     for table_name, rows in database_schema.items():
#         st.write(table_name)
#         for row in rows:
#             st.write({row})

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
You are an AI assistant that can answer questions about a selected database.
You are a helpful assistant that helps user to find information from selected database.
The database contains information in tables.
All the Tables: {all_tables}
Your task is to answer the user's query based on the provided database information.
Your persona is polite, friendly and helpful.

While Trying to solve the customer's query, you can use the following information:
 - You can ask clarifying questions to understand the user's needs better.
 - You can use the provided database that match the user's request.
 - You can provide additional information about the query.
 - You can suggest alternatives or modifications to the information based on the user's preferences.

User Query: {user_query}

Generate a SQL query to retrieve the requested information from the database.
database: sqlite:///{selected_option}
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

user_queries =  [
    #f"Who are top 5 customers ?",
    #f"Get me the details of the two oldest employees along with their age.",
    f"Give me the Product name which has highest orders and show the total orders of this Product."
]
# st.text_area("Your question for search the database:", placeholder="The database contains information about Categories, CustomerCustomerDemo, CustomerDemographics, Customers, Employees, EmployeeTerritories, Order Details, Orders, Products, Regions, Shippers, Suppliers, and Territories.", height=100)
sql_queries = []
info_name = ""
for user_query in user_queries:
    resp = model.generate_content(
        prompt.format(user_query=user_query, database_schema=database_schema, all_tables=all_tables, selected_option=selected_option) + special_instructions
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
        st.write(db_response)
        db_response = db_response.strip().split('```sql')[1].split(
        '```')[0].strip()
        info_name = pd.read_sql(db_response, engine)
        st.write(info_name)
        st.markdown("---")


embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")

stores = {}

### Commenting for other functions Please uncomment below for Vector Embeddings to work
# for table_name in tqdm(all_tables):
#     data = pd.read_sql(
#         f'''
#         SELECT * FROM "{table_name}" Limit 3
#         ''',
#         engine
#     )
#     # st.subheader(table_name)
#     # print(data.head().to_markdown())
#     # st.write(data.head().to_markdown())

#     data = [str(i) for i in data.to_dict('records')]

#     connection = SQLiteVec.create_connection(db_file=f"./vectorDB/{table_name}_vec.db")

#     vector_store = SQLiteVec(
#         table="intelligence",
#         embedding=embedding_function,
#         connection=connection,
#         )

#     _ = vector_store.add_texts(texts=data)
#     stores[table_name] = vector_store
    
# def binary_to_bytes(doc_dict):
    
#     # Extract and decode the picture binary
#     doc_dict = json.loads(doc_dict)
#     picture_data = doc_dict.get("Picture")
#     # Convert the escaped binary string into bytes
#     # This assumes Picture was serialized as base64 or raw bytes string
    
#     try:
#         # If stored as raw binary string (escaped like b'\xff\xd8...'), use eval
#         if isinstance(picture_data, str) and picture_data.startswith("b'"):
#             # Convert string like: "b'\\xff\\xd8...'" → bytes
#             picture_bytes = eval(picture_data)  # ⚠️ Safe here ONLY because you control the data
#             return st.image(BytesIO(picture_bytes), caption="Decoded Image")
#         else:
#             st.warning("Picture field not in expected format.")

#     except Exception as e:
#         st.error(f"Error decoding image: {e}")
    

# vector_store = stores['Categories']
# query = 'Picture'

# for doc in vector_store.similarity_search_with_score(query,):
    
#     json_str = json.dumps(doc[0].page_content.replace("'", '"'))
#     doc_dict = json.loads(json_str)
#     #picture_bytes = binary_to_bytes(doc_dict)
    
#     # Convert to a DataFrame
#     df = pd.DataFrame([doc_dict])  # wrap in list to make it a row
#     st.markdown("---")
#     st.dataframe(df, hide_index=True, column_config={"Picture":None})
#    # Show the image in Streamlit
#    # st.image(picture_bytes)

user_rag_queries =  [
     f"Which category does the '{info_name}' and Give me all the suppliers for that product ?"
]

rag_prompt = '''
You are an AI assistant that can answer questions about a selected database.
You are a helpful assistant that helps user to find information from selected database.
The database contains information in tables.
All the Tables: {all_tables}
Your task is to answer the user's query based on the provided database information.

User Query: {user_query}

SQL Query: {resp}

Database Response:
{data_response}

'''
st.write('-' * 26 + 'Multi-Table Queries'+'-' * 40)
for user_query in user_rag_queries:
    resp = model.generate_content(prompt.format(
        user_query=user_query, database_schema=database_schema, all_tables=all_tables, selected_option=selected_option)).text
    st.write(resp)
    resp = resp.strip().split('```sql')[1].split(
        '```')[0].strip()  # Extract SQL query from response

    try:
        
        data_response = pd.read_sql(resp, engine).to_markdown(index=False)
        
        llm_resp = model.generate_content(rag_prompt.format(
            user_query=user_query,
            resp=resp,
            data_response=data_response
        )
        ).text.strip()        

        st.write(user_query, ' : ')
        st.write(llm_resp)
        
    except Exception as e:
        print(f"Error executing query: {e}")
