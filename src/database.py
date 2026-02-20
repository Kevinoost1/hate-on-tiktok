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

Contains all database methods used for the pipeline
- Supply the path to your .sqlite3 database in a separate file (.env or .txt) when initializing the class or pass it directly

It contains the following general methods:

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
    def __init__(self, transcriptTableName, db_path: str=BASE_DIR):
        self.BASE_DIR: str = BASE_DIR 
        self.db_path: str = self.BASE_DIR / 'database' / 'db.sqlite3'
        self.transcriptTableName: str = transcriptTableName

        self.enforce_correct_structure(self.transcriptTableName)

    def connect(self, path: str, tries=10) -> sqlite3.Connection:
        if tries == 0:
            raise Exception('Failed to connect after 10 tries')

        try:
            return sqlite3.connect(path)
        except:
            time.sleep(60)
            return self.connect(path, tries-1)

    def enforce_correct_structure(self, tableName: str) -> None:
        requiredCols = {'translation': 'TEXT', 'filtered': 'BOOLEAN'}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f'PRAGMA table_info({tableName})') 
            existingCols = [row[1] for row in cursor.fetchall()]
            print(existingCols)

        for col, type in zip(requiredCols.keys(), requiredCols.values()):
            if col not in existingCols:
                cursor.execute(f"ALTER TABLE {tableName} ADD {col} {type};")

    def insert_data(self, tableName: str, data: pd.DataFrame) -> None:
        colNames: str = ', '.join(data.columns)
        placeholder: str = ', '.join(['?'] * len(data.columns))
        values = [tuple(row) for row in data.to_numpy()]

        with self.connect(self.db_path) as conn:
            query = f'''
            INSERT INTO {tableName} ({colNames}) VALUES ({placeholder})
            '''
            conn.executemany(query, values)

    def update_data(self, tableName: str, data: pd.DataFrame) -> None:
        cols = ", ".join(f"{col} = ?" for col in data.iloc[:, 1:].columns)
        values = [
            tuple(row[1:]) + (row[0],)
            for row in data.to_numpy()
        ]

        with self.connect(self.db_path) as conn:
            query = f"""
            UPDATE {tableName} SET {cols} WHERE id = ?
            """
            conn.executemany(query, values)

    def retrieve_data(self, tableName: str, colNames: list, restrictions: str=None, batch_size: int=None) -> pd.DataFrame:
        colNames: str = ', '.join(colNames)

        with self.connect(self.db_path) as conn:
            query: str = f'SELECT {colNames} FROM {tableName}'
            query += f' WHERE {restrictions}' if restrictions else ''
            query += f' LIMIT {batch_size}' if batch_size else ''

            return pd.read_sql(query, conn)

    def create_backup(self):
        date = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy(self.db_path, self.BASE_DIR / 'database' / f'db_backup_{date}.sqlite3')
        print('Database copied')




