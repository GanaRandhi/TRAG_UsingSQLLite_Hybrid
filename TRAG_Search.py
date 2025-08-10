# ==============================================================================
# IMPORTS
# ==============================================================================
# Standard library imports
from asyncio.windows_events import NULL
import os
import json
import subprocess
import sqlite3

# Third-party imports
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from sqlalchemy import create_engine
# from io import BytesIO # Commented out in original, keeping as is
# import base64 # Commented out in original, keeping as is
from langchain_community.embeddings import OllamaEmbeddings # Imported but not used directly. Keeping as is.
from langchain_community.vectorstores import SQLiteVec
from tqdm.auto import tqdm
from langchain_huggingface import HuggingFaceEmbeddings

# ==============================================================================
# PIP INSTALLS AND ENVIRONMENT SETUP
# ==============================================================================

# This will run the pip upgrade command during deployment.
# Note: Use with caution and ensure it's necessary for your specific setup.
result = subprocess.run(["pip", "install", "--upgrade", "pip"], capture_output=True, text=True)

# Load environment variables from a .env file
load_dotenv()

# Configure Google Generative AI with the API key from environment variables
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Raise an error if the API key is not found, as it's a critical dependency
        raise ValueError("GOOGLE_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    print("Google Generative AI configured successfully.")
except Exception as e:
    print(f"ERROR: Failed to configure Google Generative AI: {e}")
    print("Please ensure GOOGLE_API_KEY is set in your .env file or environment.")
    # Exit the application if the API key is missing to prevent further errors
    exit()

# Define the Generative AI model name to be used
LLM_MODEL_NAME = 'gemini-2.5-flash' # Or 'gemini-pro', 'gemini-2.5-pro' depending on needs
# Initialize the GenerativeModel
model = genai.GenerativeModel(LLM_MODEL_NAME)

# ==============================================================================
# STREAMLIT SESSION STATE INITIALIZATION
# ==============================================================================
# Initialize session state variables to control visibility of output blocks.
# These variables ensure that only one output block is visible at a time.
if 'show_sql_query_output' not in st.session_state:
    st.session_state.show_sql_query_output = False
if 'show_rag_output' not in st.session_state:
    st.session_state.show_rag_output = False
if 'show_vector_search_output' not in st.session_state:
    st.session_state.show_vector_search_output = False

# ==============================================================================
# DATABASE SELECTION AND CONNECTION
# ==============================================================================

# List available database files from the 'dist' directory
options = os.listdir("./dist/")
# Streamlit selectbox for the user to choose a database file
selected_option = st.selectbox("Select an DB:", options, index=None)

# Display the selected database to the user
if selected_option:
    st.write(f"You Selected the Database: {selected_option}")
# Construct the full path to the chosen database file
database_file = "./dist/" + str(selected_option) #@param

def get_db_connection():
    """
    Establishes and returns a SQLite database connection.
    Configures row_factory to sqlite3.Row for dictionary-like access to rows.
    """
    # Use 'with' statement for automatic connection closing
    with sqlite3.connect(database_file) as conn:
        conn.row_factory = sqlite3.Row
        return conn

def create_dbs():
    """
    Executes SQL script from 'sql/create.sql' to set up database tables.
    """
    # Get a database connection
    conn = get_db_connection()
    cursor = conn.cursor()
    # Read the SQL script from file
    with open("./sql/create.sql", 'r') as sql_file:
        sql_script = sql_file.read()
    # Execute the entire script
    cursor.executescript(sql_script)
    st.write("Database created and script executed successfully.")
    # Close cursor and connection (though 'with' handles conn, explicit close is good practice for cursor)
    cursor.close()
    conn.close()

# Initial database connection check and creation if needed
try:
    if os.path.isfile(database_file):
        print(f"The database exists.")
    else:
        # If database file does not exist, create it
        create_dbs()
except sqlite3.Error as e:
    st.write(f"An error occurred during database connection/creation: {e}")

# ==============================================================================
# DATABASE SCHEMA AND DATA RETRIEVAL
# ==============================================================================

def get_all_tables_data():
    """
    Retrieves all table names and their complete data from the connected SQLite database.
    Returns:
        tuple: A tuple containing:
            - list: A list of all table names (str).
            - dict: A dictionary where keys are table names and values are
                    pandas DataFrames containing the entire table's data.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    # Query to get all table names from sqlite_master
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    table_names = [table[0] for table in tables]

    all_tables_data = {}
    try:
        for table_name in table_names:
            # Read each table's entire content into a Pandas DataFrame
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
            all_tables_data[table_name] = df
    except sqlite3.Error as e:
        st.write(f"An error occurred while fetching table data: {e}")

    cursor.close()
    conn.close()
    return table_names, all_tables_data

def get_schema(table_name: str, formatted_string: str, schema: dict) -> tuple[str, dict]:
    """
    Retrieves the schema (column names and types) of a specified table.
    Appends the formatted schema to a string and populates a dictionary.

    Args:
        table_name (str): The name of the table to retrieve schema for.
        formatted_string (str): The string to which the formatted schema will be appended.
        schema (dict): The dictionary to store the table schema (mutated directly).

    Returns:
        tuple: A tuple containing the updated formatted string and the updated schema dictionary.
    """
    conn = get_db_connection()
    # Ensure the table name exists as a key in the schema dictionary
    schema.setdefault(table_name, [])
    # Get column information using PRAGMA table_info
    columns_info = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    # Extract column names and types
    table_columns = [(col[1], col[2]) for col in columns_info]

    # Format the schema into a readable string
    formatted_string += f"{table_name}: \n"
    for col_name, col_type in table_columns:
        formatted_string += f" {col_name} ({col_type})\n" # Added newline for better formatting
        schema[table_name].append(f"{col_name} ({col_type})")

    conn.close() # Close connection after use
    return formatted_string, schema

# Fetch all table names and their data
all_tables, all_tables_data = get_all_tables_data()

# Initialize variables for database schema
db_schema = ""
database_schema = {}

# Populate the database schema if tables are found
if all_tables:
    for table_name in all_tables:
        _, database_schema = get_schema(table_name, db_schema, database_schema)
    # st.write(f"DB Schema in text: {db_schema}") # Commented out as in original

# region Display All Table Names, Tabular Data, Database Schema
### Display sections for All Table Names, Tabular Data, Database Schema (kept commented out as requested)
if all_tables:
    st.subheader("Available Tables Names:")
    # Iterating from the second element assuming the first is not relevant or handled differently
    for table_name in all_tables[1:]:
        st.write(f"**{table_name}**")
else:
    st.info("No tables found in the database.")

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

# endregion

# ==============================================================================
# region LANGUAGE MODEL PROMPTS DEFINITION
# ==============================================================================

# Create a SQLAlchemy engine for connecting to the SQLite database
engine = create_engine(f'sqlite:///{database_file}')

# Regular expression pattern for cleaning strings (currently not used explicitly but defined)
CLEANING_PATTERN = r'[^a-zA-Z0-9]'

# Prompt template for instructing the LLM to generate SQL queries
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

# Special instructions to deliberately introduce an error in the generated SQL query
special_instructions = '''

SPECIAL INSTRUCTIONS:

MAKE AN ERROR IN THE SQL FORMATION, I WANT TO TEST ERROR HANDLING THE SQL QUERY
'''

# Prompt template for handling errors that occur during SQL query execution
error_handling_prompt = '''
You are a database response handler for business users.
Understand the user query, the generated sql and the error that has occurred.

If you can fix the query, provide the fixed query and query alone. Add in a comment: `--fixed_query` at the end to mark fix
Else, provide a polite way to say that there was an error in the query and apologize.
Your response may be seen by LLMs further down the line.

User Query: {user_query}

SQL Query: {sql}

Schema: {database_schema}

Error: {error}
'''

# Prompt template for Retrieval-Augmented Generation (RAG) to provide a human-like answer
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

# Prompt template for Retrieval-Augmented Generation (RAG) using Advanced and Complex Aggregations & Ranking
table_choice_prompt = '''

You are an experienced Engineer and have been given the task of being the data retriever.
Given the user query, make a choice on the tables to be used to fetch the data and then be able to build the answer.

user query: {user_query}

table names: {all_tables}

Respond with the table names to be used as a json list of table names
'''
r_prompt= '''
You are an AI assistant that can answer questions about a movie database.
The database contains information in tables.
All the Tables: {all_tables}
Your task is to answer the user's query based on the provided database information.

User Query: {user_query}

Generate a SQL query to retrieve the requested information from the database.
database: sqlite:///{selected_option}

database schema:
{database_schema}
'''
ag_prompt = '''
You are an AI assistant that can answer questions about a movie database.
The database contains information in tables.
All the Tables: {all_tables}

Your task is to answer the user's query based on the provided database information.

User Query: {user_query}

Chat History:
{chat_history}

Database Response:
{data_response}
'''
# endregion

# ==============================================================================
# region SQL QUERY GENERATION AND EXECUTION UI
# ==============================================================================

# Streamlit text area for the user to input their query for the database
user_query = st.text_area("Your question for search the database:", height=100)
# Example queries (commented out in original, keeping as is)
# [
#     #f"Who are top 5 customers ?",
#     #f"Get me the details of the two oldest employees along with their age.",
#     f"Give me the Product name which has highest orders and show the total orders of this Product."
#]

# List to store SQL queries (for logging or potential future use)
sql_queries = []
# Variable to hold information name (used in error handling path)
info_name = ""

# Button to trigger the SQL query generation and execution process
# When this button is clicked, it will show its output and hide others.
if st.button("Generate SQL and Query Database", key="button_sql_query"):
    st.session_state.show_sql_query_output = True
    st.session_state.show_vector_search_output = False
    st.session_state.show_rag_output = False
    st.session_state.show_adv_cmplx_rag_output = False
    if user_query: # Ensure there's a user query to process
        # Generate the SQL query using the LLM, including special instructions for error testing
        resp = model.generate_content(
            prompt.format(user_query=user_query, database_schema=database_schema, all_tables=all_tables, selected_option=selected_option) + special_instructions
            ).text
        # Extract the SQL query string from the LLM's response
        sql = resp.strip().split('```sql')[1].split('```')[0].strip()
        sql_queries.append(sql) # Add the generated SQL to the list
        st.markdown("---") # Separator for UI clarity
        print(sql, '\n\n') # Print the SQL query to console for debugging

        db_response = None
        try:
            # Attempt to execute the generated SQL query using pandas
            db_response = pd.read_sql(sql, engine).to_markdown(index=False)
            st.write(db_response) # Display the database response
        except Exception as e:
            # If an error occurs during SQL execution, use the error handling prompt
            db_response = model.generate_content(
                error_handling_prompt.format(
                    error=str(e), # Pass the error message
                    user_query=user_query,
                    sql=sql,
                    database_schema=database_schema
                  )
              ).text
            st.write(db_response) # Display the LLM's error handling response
            # Attempt to extract a fixed SQL query from the error handling response
            # This assumes the LLM correctly provided a fixed query within ```sql tags
            try:
                fixed_sql = db_response.strip().split('```sql')[1].split('```')[0].strip()
                # Execute the fixed query
                info_name = pd.read_sql(fixed_sql, engine)
                st.write("Fixed Query Result:")
                st.write(info_name)
            except Exception as fix_e:
                st.write(f"Could not extract or execute fixed query: {fix_e}")
                st.write("Please check the generated error handling response.")
        st.markdown("---") # Another UI separator
        
# endregion

# ==============================================================================
# region # VECTOR SEARCH AND EMBEDDING GENERATION
# ==============================================================================

# def check_existence(filename_to_check: str) -> bool:
#     """
#     Checks if a given filename exists within the './vectorDB/' directory.

#     Args:
#         filename_to_check (str): The specific filename to look for.

#     Returns:
#         bool: True if the filename exists, False otherwise.
#     """
#     folder_path = "./vectorDB/"
#     try:
#         # Get all entries (files and directories) in the specified folder
#         all_entries = os.listdir(folder_path)
#         # Filter the entries to include only actual files
#         filenames = [f for f in all_entries if os.path.isfile(os.path.join(folder_path, f))]
#         # Check if the specified filename is in the list of files
#         file_exists = filename_to_check in filenames
#         return file_exists
#     except FileNotFoundError:
#         print(f"Error: Folder not found at '{folder_path}'")
#         return False
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return False

# # Initialize the embedding function using a pre-trained HuggingFace model
# embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/LaBSE")
# # Dictionary to store vector store instances for each table
# stores = {}

# # Iterate through all fetched table data to create or load vector embeddings
# for table_name, df in tqdm(all_tables_data.items()):
#     data = df
#     # Convert DataFrame rows to string representations for embedding
#     data_records = [str(i) for i in data.to_dict('records')]
#     # Define the vector database file name for the current table
#     vec_db_filename = f"{table_name}_vec.db"

#     # Check if the vector database for the current table already exists
#     if not check_existence(vec_db_filename):
#         st.write(f"Please wait while the table '{table_name}' is being Vector Embedded....")
#         # Create a new SQLite connection for the vector store
#         connection = SQLiteVec.create_connection(db_file=f"./vectorDB/{vec_db_filename}")

#         # Initialize the SQLiteVec vector store
#         vector_store = SQLiteVec(
#             table="intelligence", # Table name within the SQLiteVec database
#             embedding=embedding_function,
#             connection=connection,
#         )
#         # Add the text data to the vector store
#         _ = vector_store.add_texts(texts=data_records)
#         # Store the vector store instance in the 'stores' dictionary
#         stores[table_name] = vector_store
#     else:
#         # If the vector DB already exists, load it
#         connection = SQLiteVec.create_connection(db_file=f"./vectorDB/{vec_db_filename}")
#         vector_store = SQLiteVec(
#             table="intelligence",
#             embedding=embedding_function,
#             connection=connection,
#         )
#         stores[table_name] = vector_store
#         print(f"Vector DB for '{table_name}' loaded from existing file.")


# # Streamlit UI for performing vector search
# if selected_option: # Ensure a database option is selected
#     # Dropdown to allow the user to pick a table for vector search
#     vector_store_selection_name = st.selectbox("Pick a Table for Vector Search:", all_tables, index=None)
#     # Text area for the user to input their vector search query
#     query_vector_search = st.text_area("Enter your Vector Search Word:", key="vector_search_query")

#     # Button to trigger the vector search
# if st.button("Vector Search Me", key="button_vector_search"):
        # st.session_state.show_vector_search_output = True
        # st.session_state.show_sql_query_output = False
        # st.session_state.show_rag_output = False
        # st.session_state.show_adv_cmplx_rag_output = False
#         if vector_store_selection_name and query_vector_search:
#             # Retrieve the appropriate vector store instance based on user selection
#             selected_vector_store = stores.get(vector_store_selection_name)
#             if selected_vector_store:
#                 st.write(f"Performing vector search in '{vector_store_selection_name}' for: '{query_vector_search}'")
#                 # Perform similarity search with score
#                 results = selected_vector_store.similarity_search_with_score(str(query_vector_search))
#                 if results:
#                     for doc, score in results:
#                         # Parse the document content (which is a string representation of a dictionary)
#                         json_str = json.dumps(doc.page_content.replace("'", '"'))
#                         doc_dict = json.loads(json_str)
#                         # Convert the document dictionary to a Pandas DataFrame for display
#                         df_result = pd.DataFrame([doc_dict])
#                         st.markdown("---") # UI separator
#                         st.dataframe(df_result, hide_index=True) # Display the result
#                         st.write(f"Similarity Score: {score:.4f}") # Display similarity score
#                 else:
#                     st.info("No matching results found in vector store.")
#             else:
#                 st.warning("Please select a valid table for vector search.")
#         else:
#             st.warning("Please select a table and enter a query for vector search.")

# endregion

# ==============================================================================
# region MULTI-TABLE RETRIEVAL-AUGMENTED GENERATION (RAG)
# ==============================================================================

# Streamlit text area for a follow-up question, often used in a RAG context
user_query_for_rag = st.text_area("Your question for a followup question above:", height=100, key="rag_followup_query")
# Example RAG queries (commented out in original, keeping as is)
# [
#      f"Which category does the '{info_name}' along with Give results all the suppliers for that product ?"
# ]

# Button to trigger the Multi-Table RAG process
# When this button is clicked, it will show its output and hide others.
if st.button("Multi-Table RAG Search", key="button_multi_tbl_rag"):
    st.session_state.show_rag_output = True
    st.session_state.show_sql_query_output = False
    st.session_state.show_vector_search_output = False
    st.session_state.show_adv_cmplx_rag_output = False
    st.write('-' * 26 + 'Multi-Table RAG'+'-' * 40) # UI header
    if user_query_for_rag: # Ensure there's a follow-up query
        # Generate a new SQL query based on the RAG query using the main prompt
        resp_llm_sql = model.generate_content(prompt.format(
            user_query=user_query_for_rag,
            database_schema=database_schema,
            all_tables=all_tables,
            selected_option=selected_option
        )).text
        st.write("Generated SQL by LLM:")
        st.write(resp_llm_sql) # Display the LLM's full response (including SQL)
        # Extract the pure SQL query string
        generated_sql = resp_llm_sql.strip().split('```sql')[1].split('```')[0].strip()

        try:
            # Execute the generated SQL query to get data from the database
            data_response_df = pd.read_sql(generated_sql, engine)
            data_response_markdown = data_response_df.to_markdown(index=False)
            st.write("Database Response:")
            st.write(data_response_markdown) # Display the raw database response

            # Use the RAG prompt to generate a human-friendly answer based on the query, SQL, and data response
            final_llm_response = model.generate_content(rag_prompt.format(
                user_query=user_query_for_rag,
                resp=generated_sql, # Pass the generated SQL query
                data_response=data_response_markdown # Pass the database response in markdown
            )).text.strip()

            st.write(f"Your Question: {user_query_for_rag}")
            st.write("AI Assistant's Answer:")
            st.write(final_llm_response) # Display the final human-readable answer
            st.markdown("---") # UI separator

        except Exception as e:
            st.error(f"Error during RAG process: {e}")
            st.write(f"Failed SQL Query: `{generated_sql}`")
#endregion

# ===================================================================================
# region Advanced and Complex Aggregations & Ranking RETRIEVAL-AUGMENTED GENERATION (RAG)
# ===================================================================================
def tables_json():
    
    table_choice = model.generate_content(table_choice_prompt).text
    json_parsed = json.loads(table_choice.split('```json')[-1].split('```')[0])
    return (json_parsed)

def build_ddl_prompt(json_parsed):
  schemas = []
  dbs={}
  for table_name in json_parsed:
      schema_ , _ = get_schema(table_name,"", dbs)
    #   print(schema_)
      schemas.append(schema_)

  schemas_prompt = '\n\n'.join(schemas)
#   print(schemas)
  return (schemas_prompt)

def r_of_rag(user_query, schemas_prompt):

  resp = model.generate_content(r_prompt.format(
    user_query=user_query, 
    database_schema=schemas_prompt, 
    all_tables = all_tables,    
    selected_option=selected_option
    )).text
  try:
    st.write(resp)
    sql = resp.strip().split('```sql')[1].split(
      '```')[0].strip()  # Extract SQL query from response
    data_response = pd.read_sql(sql, engine).to_markdown(index=False)
    return data_response
  except Exception as e:
    print(e)
    return ""

def ag_of_rag(user_query, data_response, chat_history):
    try:
        llm_resp = model.generate_content(ag_prompt.format(
            user_query=user_query,
            data_response=data_response,
            chat_history=chat_history,
            all_tables=all_tables
        )).text.strip()  
        return llm_resp 
    except Exception as e:
        st.write(f"Error executing query: {e}")

# Streamlit text area for a follow-up question, often used in a RAG context
user_input_adv_rag = st.text_area("Your question for a Adv & Complex questions(shift=enter for multi lines):", height=150, key="rag_adv_cmplx_query")
# Example RAG queries (commented out in original, keeping as is)
# [
#      f'I need to understand the popularity of the movie {info_name}',
#      f'What are the movies in the similar popularity scale or higher?',
#      f'What are the most common Genres for these movies?',
#      f'Ok, I want to understand what could be a common cast to such successful movies. Rank them by their performance in terms of overall movie rating'
# ]

# Button to trigger the Multi-Table RAG process
# When this button is clicked, it will show its output and hide others.
if st.button("Adv & Complex Aggr RAG", key="button_adv_complex_rag"):
    st.session_state.show_adv_cmplx_rag_output = True
    st.session_state.show_rag_output = False
    st.session_state.show_sql_query_output = False
    st.session_state.show_vector_search_output = False
# Check if there's any input
    user_queries =[]
    history = []
    if user_input_adv_rag:
        # Split the input string into a list of lines
        user_queries = user_input_adv_rag.splitlines()
        # for line in lines:
        #     st.write(f"- {line}")

    for user_query in user_queries:
        json_parsed = tables_json()
        schemas_prompt = build_ddl_prompt(json_parsed)
        data_response = r_of_rag(user_query,schemas_prompt)
        #st.write(f'R of RAG - Data Response : {data_response}')
        history.append(f'data_context: \n{data_response}')
        llm_resp = ag_of_rag(user_query,data_response,history)
        history.append(f'user_query: {user_query}\nanswer: {llm_resp}')
        #st.write(user_query, ':', llm_resp)
        [st.write(msg, '\n', '-'*80) for i, msg in enumerate(history) if i%2==1]
# endregion