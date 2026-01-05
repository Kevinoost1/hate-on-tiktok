import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, MBartForConditionalGeneration, MBart50Tokenizer
import torch
import re
import pandas as pd
import shutil
from datetime import datetime
import time
from torch.cuda.amp import autocast
from collections import defaultdict


class preprocess:
    def __init__(self):
        self.dbPath: str = ''
        self.batchSize: int = 99
        self.device: str = 'cuda'
        self.translationModel: str = 'facebook_translator'
        self.translationLanguage: str = 'nl_XX'

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.dbPath)

    def _insertData(self, df: pd.DataFrame, table: str, idColumn: str, *columns: str):
        columnsToUpdate = ", ".join(f"{col} = ?" for col in columns)

        query = f"""
        UPDATE {table}
        SET {columnsToUpdate}
        WHERE {idColumn} = ?
        """
        params = [
            tuple(row[col] for col in columns) + (row[idColumn],) for _, row in df.iterrows()
        ]
        return query, params


    def retrieveBatchFromDb(self) -> pd.DataFrame:
        with self._connect() as db:
            query = f"""
            SELECT id, video_id, language_id, transcript, filtered, translation
            FROM transcript_classification
            WHERE translation IS NULL
            AND (language_id IN (2, 18))
            ORDER BY id ASC
            LIMIT {self.batchSize}
            """
            return pd.read_sql(query, db)

    def preprocessText(self):
        with sqlite3.connect(self.dbPath) as db:
            query = f"""
            SELECT id, content
            FROM subtitles
            """
            rawSubtitles = pd.read_sql(query, db)
            rawSubtitles['transcript'] = rawSubtitles['content'].apply(self.textMatching)
            db.cursor.executemany(self._insertData(rawSubtitles, 'transcript_classification', 'id', 'transcript'))
            db.commit()

    @staticmethod
    def textMatching(transcript: str) -> str:
        newText = ''
        for word in transcript.split():
            if re.match(r"\b(\d{2}):(\d{2}):(\d{2})\.(\d{3})\b", word):
                continue
            match word:
                case 'WEBVTT' | '-->':
                    continue
                case _:
                    newText += f"{word} "
        return newText




