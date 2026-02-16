import sqlite3
import pandas as pd


ENV_PATH = r'C:\Users\2mrwm\Documents\phd\project_classification\env\database.env'

class Database:
    def __init__(self, env_fileName: str = ENV_PATH):
        self.env_fileName: str = env_fileName
        self.db_path: str = self.read_env()

    def read_env(self) -> str:
        with open(self.env_fileName, 'r') as f:
            return f.read()

    def connect(self, path: str) -> sqlite3.Connection:
        return sqlite3.connect(path)

    def insert_data(self, tableName: str, data: dict) -> None:
        colNames: str = ', '.join(data.keys())
        values = data.values()
        placeholder: str = ', '.join(['?'] * len(values))

        with self.connect(self.db_path) as conn:
            query = f'''
            INSERT INTO {tableName} ({colNames}) VALUES ({placeholder})
            '''

            conn.execute(query)

    def retrieve_batch(self, batch_size: int, tableName: str, colNames: list, restrictions: str) -> pd.DataFrame:
        colNames: str = ', '.join(colNames)

        with self.connect(self.db_path) as conn:
            query: str = f'''
            SELECT {colNames} FROM {tableName}
            WHERE {restrictions}
            LIMIT {batch_size}
            '''

            return pd.read_sql(query, conn)



