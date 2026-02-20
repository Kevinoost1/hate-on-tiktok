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
import json
import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

"""
In main.py you can find an example use for the entire pipeline

*The first step in the classification pipeline*
This class works with pandas dataframes, with .sqlite3 database methods provided seperately in database.py
Below is a schematic of the preprocessing steps:

preprocess_text -> generate_translation (Dutch to English) -> filter_on_heuristics 

preprocess_text() gets rid of all artifacts related to the scraping (e.g. timestamps, json parsing)
- It requires a pandas dataframe and the column name for the raw text transcripts. 
- When finished this function return a dataframe with the data in the raw transcripts column modified to keep only the actual text.

generate_translation() returns a dataframe with a new column 'translation'. 
- It requires a pandas dataframe, the column name for the transcripts, and the column name for the language identifiers. 
- Only the Dutch transcripts are translated into English. The transcripts where the original language was English are passed as is.
Transcripts in any other language are given the label "OTHER_LANG"
- Some transcripts are too long to pass into the model directly. Whenever this is the case, the transcript is split until the last allowed word.
The parts are then translated seperately and merged.
- This method has one hyperparameter associated with it, namely the batch size. 
Depending on your setup (mainly GPU memory), you can set this higher or lower. Generally, higher batch sizes make the translation process faster, but require more VRAM in the GPU.
- The 'translation' column in the new dataframe matches with an empty database column, created upon initialization of the Database() class and thus requires nothing further.

filter_on_heuristics() returns a dataframe with a new column 'filtered' (0 or 1)
- It requires a pandas dataframe and the column name for the transcripts.
- Whenever a transcript is less than 8 words or has a word diversity of less than 0.3 (unique words/total words), it has a value of 1 in filtered.
These hyperparameters can off course be changed within the class
- As with the translation, the Database() initialization should have already created an empty 'filtered' column if it did not exist yet. 
"""
class Preprocess:
    def __init__(self):
        self.device: str = 'cuda'

        self.translationModelName: str = BASE_DIR / 'models' / 'facebook_translator' 
        self.translationBatchSize: int = 16

        self.maxWordsAllowed: int = 8
        self.maxDiversityAllowed: float = 0.3

    def preprocess_text(self, df: pd.DataFrame, subtitlesColumn:str) -> pd.DataFrame:
        df[subtitlesColumn] = df[subtitlesColumn].apply(self.match_text)
        return df

    def match_text(self, transcript: str) -> str:
        if re.match(r'{"utterances":', transcript):
            transcript = self.clean_json_utterances(transcript)

        newText: str = ''
        for word in transcript.split():
            if re.match(r"\b(\d{2}):(\d{2}):(\d{2})\.(\d{3})\b", word):
                continue
            match word:
                case 'WEBVTT' | '-->':
                    continue
                case _:
                    newText += f"{word} "
        return newText

    @staticmethod
    def clean_json_utterances(text: str) -> str:
        obj = json.loads(text)
        utt = obj.get("utterances", [])
        parts = [u.get("text", "").strip() for u in utt if isinstance(u, dict)]
        return " ".join([p for p in parts if p]).strip()

    def generate_translation(self, df: pd.DataFrame, transcriptColumn:str, languageIdColumn: str) -> pd.DataFrame:
        indexes: [int] = df.index.tolist()
        transcripts: [str] = df[transcriptColumn].tolist()
        language_ids: [int] = df[languageIdColumn].tolist()

        parts: [str] = []
        mapping: [int] = []

        for i, (idx, transcript, langId) in enumerate(zip(indexes, transcripts, language_ids)):
            if langId == 18:
                for part in self.split_text(transcript):
                    parts.append(part)
                    mapping.append(idx)
            elif langId == 2:
                df.at[idx, 'translation'] = transcript
            else:
                df.at[idx, 'translation'] = 'OTHER_LANG"'

        model = MBartForConditionalGeneration.from_pretrained(self.translationModelName).to(self.device).eval()
        tokenizer = MBart50Tokenizer.from_pretrained(self.translationModelName, src_lang="nl_XX", tgt_lang="eng_XX")
        decoded: list = [None] * len(parts)

        for start in range(0, len(parts), self.translationBatchSize):
            batch: [str] = parts[start:start + self.translationBatchSize]

            tokenized: {int: torch.tensor} = tokenizer(batch, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                with autocast(dtype=torch.float16):
                    output: {int: torch.tensor} = model.generate(**tokenized, num_beams=1)

            decoded_batch: [str] = tokenizer.batch_decode(output, skip_special_tokens=True)
            for i, transcript in enumerate(decoded_batch):
                decoded[start + i] = transcript.strip()

            accum = defaultdict(list)
            for part_idx, label in enumerate(mapping):
                text_part: str = decoded[part_idx] or ''
                accum[label].append(text_part)

            for label, parts_list in accum.items():
                joined_text: str = ' '.join(p for p in parts_list if p)
                df.at[label, 'translation'] = joined_text

            return df

    @staticmethod
    def split_text(transcript: str, maxLength: int=750) -> [str]:
        if len(transcript) <= maxLength:
            return [transcript]

        parts: list = []
        while len(transcript) > maxLength:
            split: [str] = transcript[:maxLength].rfind(' ')

            parts.append(transcript[:split + 1].strip())
            transcript: str = transcript[split + 1:].strip()

        if transcript:
            parts.append(transcript)

        return parts

    def filter_on_heuristics(self, df, transcriptColumn: str) -> pd.DataFrame:
        transcripts: pd.Series = df[transcriptColumn].str.lower().str.split()
        transcriptLengths: pd.Series = transcripts.str.len()

        wordDiversity: pd.Series = transcripts.apply(lambda x: len(set(x)) / len(x) if x else 0)
        df.loc[:, 'filtered'] = (transcriptLengths < self.maxWordsAllowed) | (wordDiversity < self.maxDiversityAllowed)

        return df




