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

        self.translationModelName: str = r"C:\Users\2mrwm\OneDrive - Universiteit Leiden (1)\facebook_translator"
        self.translationPath: str = r'C:\Users\2mrwm\OneDrive - Universiteit Leiden (1)'
        self.translationLanguage: str = 'nl_XX'
        self.translationBatchSize: int = 16

    def preprocess_text(self, df: pd.DataFrame, subtitlesColumn:str) -> pd.DataFrame:
        df[subtitlesColumn] = df[subtitlesColumn].apply(self.match_text)
        return df

    @staticmethod
    def match_text(transcript: str) -> str:
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
                df.at[label, transcriptColumn] = joined_text

            return df

    def split_text(self, transcript: str, maxLength: int=750) -> [str]:
        if len(transcript) <= maxLength:
            return [transcript]

        parts = []
        while len(transcript) > maxLength:
            split = transcript[:maxLength].rfind(' ')
            if split == -1:
                split = maxLength
            parts.append(transcript[:split + 1].strip())
            transcript = transcript[split + 1:].strip()

        if transcript:
            parts.append(transcript)

        return parts




