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


class Preprocess:
    def __init__(self):
        self.device: str = 'cuda'

        self.translationModel: str = 'facebook_translator'
        self.translationLanguage: str = 'nl_XX'
        self.translationBatchSize: int = 16
    def preprocess_text(self, df: pd.DataFrame, subtitlesColumn) -> pd.DataFrame:
        df[subtitlesColumn] = df[subtitlesColumn].apply(self.match_text)
        return df

    @staticmethod
    def match_text(transcript: str) -> str:
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




