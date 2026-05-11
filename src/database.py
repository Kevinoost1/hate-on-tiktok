import sqlite3
import pandas as pd
import time
import shutil
from datetime import datetime
import os
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

"""
In main.py you can find an example use for the entire analysis pipeline

*Contains all database methods used for the pipeline*
    - If you cloned the repo, it will automatically set your working directory to the folder above 'src'.
    You can then add, e.g., a folder 'database' to store your .sqlite3 file. 
    If your structure is different, you can adjust the global BASE_DIR variable, or pass a custom path when initializing the class.

It contains the following general methods:

- enforce structure -> you can run this to check that your database table matches the general structure that is desired for the program.
  - For example, within your transcript table, the program requires a 'translation' and 'filtered column to store the preprocesing steps.
    It also comes with a check for the following tables 'topic', 'topic_prob', 'sent', 'sent_prob'.
    This can function as a template structure for later classifications. 
    However, if you wish to change the names or structure (depending on your aim), just adjust the names in the dictionairy within the function.

- insert data -> you can use this when you obtain new data from new participants or sources. 
    This is useful when you want to import data that does not already have an associated id in the database
    - supply the table name (where you want to add the data)
    - supply the data as a pandas dataframe
    
- update data -> you can use this when you have a new version of data that is already in the database.
    - supply the table name (where you want to add the data)
    - supply the data as a pandas dataframe with id column as its first column
    
- retrieve data -> you can use this general function for retrieving data from the database as a pandas dataframe
    - supply the table name (where you want to add the data)
    - supply the names for columns that you might want to retrieve
    - Optionally specify any restrictions.
        e.g. if you want to retrieve data that has no translation yet and is not filtered you can provide:
            - translation IS NULL AND filtered = 0
    - Optionally specify a batch size for when you wish to retrieve up to a certain number of rows
    
- create a backup -> you can use this when you wish to create a timestamped backup of the database
"""

class Database:
    def __init__(self, transcriptTableName: str='transcript_classification', db_path: str=BASE_DIR / 'database' / 'db.sqlite3'):
        self.BASE_DIR: str = BASE_DIR 
        self.db_path: str = db_path
        self.transcriptTableName: str = transcriptTableName
 
    def connect(self, tries: int=10) -> sqlite3.Connection:
        if tries == 0:
            raise Exception('Failed to connect after 10 tries')

        try:
            return sqlite3.connect(self.db_path)
        except:
            time.sleep(60)
            return self.connect(tries-1)

    def enforce_correct_structure(self, tableName: str) -> None:
        requiredCols = {'translation': 'TEXT', 'filtered': 'BOOLEAN', 'topic': 'TEXT', 'topic_prob': 'REAL', 
                        'sent': 'TEXT', 'sent_prob': 'REAL', 'intolerance': 'REAL', 'intolerance_prob': 'REAL'}

        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute(f'PRAGMA table_info({tableName})') 
            existingCols = [row[1] for row in cursor.fetchall()]

            for col, type in zip(requiredCols.keys(), requiredCols.values()):
                if col not in existingCols:
                    cursor.execute(f"ALTER TABLE {tableName} ADD {col} {type};")

    def insert_data(self, tableName: str, data: pd.DataFrame) -> None:
        colNames: str = ', '.join(data.columns)
        placeholder: str = ', '.join(['?'] * len(data.columns))
        values = [tuple(row) for row in data.to_numpy()]

        with self.connect() as conn:
            query = f'''
            INSERT INTO {tableName} ({colNames}) VALUES ({placeholder})
            '''
            conn.executemany(query, values)

    def update_data(self, tableName: str, data: pd.DataFrame, idColumnName: str = 'id') -> None:
        cols_to_update = [col for col in data.columns if col != idColumnName]
        cols: str = ", ".join(f"{col} = ?" for col in cols_to_update)
        
        values = [
            tuple(row[col] for col in cols_to_update) + (int(row[idColumnName]),)
            for _, row in data.iterrows()
        ]

        with self.connect() as conn:
            query = f"""
            UPDATE {tableName} SET {cols} WHERE {idColumnName} = ?
            """
            conn.executemany(query, values)
            conn.commit()


    def retrieve_data(self, tableName: str, colNames: list[str], restrictions: str=None, orderby: str=None, batch_size: int=None, offset: int=None) -> pd.DataFrame:
        colNames: str = ', '.join(colNames)

        with self.connect() as conn:
            query: str = f'SELECT {colNames} FROM {tableName}'
            query += f' WHERE {restrictions}' if restrictions else ''
            query += f' ORDER BY {orderby}' if orderby else ''
            query += f' LIMIT {batch_size}' if batch_size else ''
            query += f' OFFSET {offset}' if offset else ''

            return pd.read_sql(query, conn)

    def create_backup(self):
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(self.db_path, self.BASE_DIR / 'database' / f'db_backup_{date}.sqlite3')
        print('Database copied')




