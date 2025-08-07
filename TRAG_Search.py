from datetime import timedelta, datetime
from random import randint
from random import choice as rc
import sqlite3
import os


# This function will return a random datetime between two datetime objects.
def random_date(start, end):
    return start + timedelta(seconds=randint(0, int((end - start).total_seconds())))

def create_update_dbs():
    with open("./sql/create.sql", 'r') as sql_file:
        sql_script = sql_file.read()
    cursor.executescript(sql_script)
    print("Database created and script executed successfully.")    
    
# Connect to the DB
try:         

    database_file = "./dist/northwind.db"
    
    with sqlite3.connect(database_file) as conn:
        cursor = conn.cursor()

    if os.path.isfile(database_file):
        print(f"The database exists.")
    else:
        create_update_dbs()
    
    cursor.execute("SELECT * FROM Products Limit 10")
    rows = cursor.fetchall()
    for row in rows:
        print(row)
    
    cursor.close()
    conn.close()
    
except sqlite3.Error as e:
    print(f"An error occurred: {e}")