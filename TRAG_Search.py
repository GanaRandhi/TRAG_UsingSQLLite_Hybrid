# ==============================================================
# 📄 TRAG_Search.py
# Interactive Streamlit App for Database Search & Vector Queries
# ==============================================================

from operator import index
import sqlite3
import os
from typing import Any, Optional
import streamlit as st
import logging
import google.generativeai as genai
import pandas as pd
import json
from asyncio.windows_events import NULL
from cmd import PROMPT
from datetime import timedelta, datetime
from random import randint, choice as rc
from dotenv import load_dotenv
from sqlalchemy import create_engine
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import SQLiteVec
from tqdm.auto import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
import subprocess

# ------------------------------
# ⚙️ Setup & Environment
# ------------------------------

# Upgrade pip during deployment (use cautiously)
result = subprocess.run(
    ["pip", "install", "--upgrade", "pip"],
    capture_output=True,
    text=True
)

# ------------------------------
# 📂 Database Selection
# ------------------------------
options = os.listdir("./dist/")
selected_option = st.selectbox("Select an DB:", options, index=None) 
if selected_option:
    st.write(f"You Selected the Database: {selected_option}")
database_file = "./dist/" + str(selected_option)

# ------------------------------
# 🔌 Database Connection Functions
# ------------------------------
def get_db_connection():
    """Create and return a connection to the selected SQLite database."""
    with sqlite3.connect(database_file) as conn:
        conn.row_factory = sqlite3.Row
        return conn

def create_dbs():
    """Create the database schema from SQL script if DB is missing."""
    with open("./sql/create.sql", 'r') as sql_file:
        sql_script = sql_file.read()
    cursor.executescript(sql_script)
    st.write("Database created and script executed successfully.")    

def get_all_tables_data():
    """Return all table names and their data."""
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

# ------------------------------
# 🛠 Connect to DB & Load Schema
# ------------------------------
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
database_schema = {}
if all_tables:
    for table_name in all_tables:
        db_schema, database_schema = get_schema(table_name, db_schema, database_schema)
    # st.write(f"DB Schema in text: {db_schema}")

# ------------------------------
# 🤖 Google Generative AI Setup
# ------------------------------
load_dotenv()
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    print("Google Generative AI configured successfully.")
except Exception as e:
    print(f"ERROR: Failed to configure Google Generative AI: {e}")
    print("Please ensure GOOGLE_API_KEY is set in your .env file or environment.")
    exit()  # Critical dependency

# ------------------------------
# 🧠 LLM Model
# ------------------------------
LLM_MODEL_NAME = 'gemini-2.5-flash'
model = genai.GenerativeModel(LLM_MODEL_NAME)
engine = create_engine(f'sqlite:///{database_file}')

# ------------------------------
# 📜 Prompt Templates
# ------------------------------
prompt = ''' ... '''  # (Keep original content)
special_instructions = ''' ... '''
error_handling_prompt = ''' ... '''

# ------------------------------
# 🔍 Query the Database
# ------------------------------
user_query = st.text_area("Your question for search the database:", height=100)
sql_queries = []
info_name = ""

if st.button("Click Me", key="button1"):
    if user_query: 
        resp = model.generate_content(
            prompt.format(
                user_query=user_query,
                database_schema=database_schema,
                all_tables=all_tables,
                selected_option=selected_option
            ) + special_instructions
        ).text
        sql = resp.strip().split('```sql')[1].split('```')[0].strip()
        sql_queries.append(sql)
        st.markdown("---")
        print(sql, '\n\n')
        try:
            db_response = pd.read_sql(sql, engine).to_markdown(index=False)
            st.write(db_response)
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
            db_response = db_response.strip().split('```sql')[1].split('```')[0].strip()
            info_name = pd.read_sql(db_response, engine)
            st.write(info_name)
        st.markdown("---")

# # ------------------------------
# # 📦 Vector Embedding Setup
# # ------------------------------
# def check_existence(filename_to_check):
#     """Check if vector DB file exists in ./vectorDB/."""
#     folder_path = "./vectorDB/"
#     try:
#         filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
#         return filename_to_check in filenames
#     except FileNotFoundError:
#         print(f"Error: Folder not found at '{folder_path}'")
#         return False
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return False

# embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
# stores = {}

# for table_name, df in tqdm(all_tables_data.items()):
#     data = [str(i) for i in df.to_dict('records')]
#     if not check_existence(str(table_name) + "_vec.db"):
#         st.write(f"Please wait while the table {table_name} is Vector Embedding....")
#         connection = SQLiteVec.create_connection(db_file=f"./vectorDB/{table_name}_vec.db")
#         vector_store = SQLiteVec(
#             table="intelligence",
#             embedding=embedding_function,
#             connection=connection,
#         )
#         _ = vector_store.add_texts(texts=data)
#         stores[table_name] = vector_store

# # ------------------------------
# # 🔍 Vector Search Section
# # ------------------------------
# if selected_option:
#     vector_store = st.selectbox("Pick a Table", all_tables, index=None)
#     query = st.text_area("Give your Vector Search Word:")

#     if st.button("Vector Search Me", key="button3"):
#         if vector_store:
#             for doc in vector_store.similarity_search_with_score(str(query),):
#                 json_str = json.dumps(doc[0].page_content.replace("'", '"'))
#                 doc_dict = json.loads(json_str)
#                 df = pd.DataFrame([doc_dict])
#                 st.markdown("---")
#                 st.dataframe(df, hide_index=True)

# ------------------------------
# 🔄 Follow-Up RAG Query
# ------------------------------
user_rag_query = st.text_area("Your question for a followup question above:", height=100)
rag_prompt = ''' ... '''  # (Keep original)

if st.button("Click Me", key="button2"):
    st.write('-' * 26 + 'Multi-Table RAG' + '-' * 40)
    if user_rag_query:
        user_query = user_rag_query
        resp = model.generate_content(
            prompt.format(
                user_query=user_query,
                database_schema=database_schema,
                all_tables=all_tables,
                selected_option=selected_option
            )
        ).text
        st.write(resp)
        resp = resp.strip().split('```sql')[1].split('```')[0].strip()
        try:
            data_response = pd.read_sql(resp, engine).to_markdown(index=False)
            st.write(data_response)
            llm_resp = model.generate_content(
                rag_prompt.format(
                    user_query=user_query,
                    resp=resp,
                    data_response=data_response
                )
            ).text.strip()
            st.write(user_query, ' : ')
            st.write(llm_resp)
            st.markdown("---")
        except Exception as e:
            print(f"Error executing query: {e}")
