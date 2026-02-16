import sqlite3
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

        query = f'''
        INSERT INTO {tableName} ({colNames}) VALUES ({placeholder})
        '''
        with self.connect(self.db_path) as conn:
            conn.execute(query)




