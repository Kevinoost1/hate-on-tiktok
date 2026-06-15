import pandas as pd
from database import *
import json
import re
from pathlib import Path
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent

class Hate_Speech_Target:
    def __init__(self):
        self.keywords = self.load_keywords()

    def main(self) -> dict[str, int]:
        df = self.get_hate_speech()

        df["text"] = (df["translation"].fillna("").str.strip() + 
            " " + df["desc"].fillna("").str.replace("#", "", regex=False).str.strip()).str.strip()

        freq = {k: 0 for k in self.keywords.keys()}

        for text in df['text'].str.lower():
            # RD: Per transcript, each group/politician appearance only counts once
            # E.g. five mentions of trump in one transcript only contributes 1 to the frequency dict
            checked_categories = set()

            for category, words in self.keywords.items():
                for word in words:
                    if word in text:
                        checked_categories.add(category)
                        break

            for cat in checked_categories:
                freq[cat] += 1
        return freq
    
    def load_keywords(self) -> dict[str, int]:
        with open(BASE_DIR/'keywords.json', 'r') as f:
            keywords = json.load(f)

        keyword_map = {}
        for broad_category, specifics in keywords['keywords'].items():
            for specific, keywords in specifics.items():
                keyword_map[specific] = [w for w in keywords]

        return keyword_map

    def get_hate_speech(self) -> pd.DataFrame:
        db = Database()
        return db.retrieve_data(
            tableName = 'transcript_classification tc LEFT JOIN videos v ON tc.video_id=v.id',
            colNames = ['tc.id', 'tc.translation', 'v.desc'],
            restrictions = 'intolerance > 5'
        ) 

class Correlation_Matrix:
    def __init__(self):
        self.db = Database()
        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}

    def get_data(self) -> pd.DataFrame:
        q =  self.db.retrieve_data(
            tableName='questionnaires', 
            colNames=['id', 'gender', 'age', 'education', 'political_preference', 'why_online1', 'why_online2', 'why_online3', 
                      'content_choice1', 'content_choice2', 'content_choice3', 'self_control1', 'self_control2', 'self_control3', 
                      'misinformation1', 'misinformation2', 'misinformation3', 'boredom1', 'boredom2', 'boredom3', 'cohort', 
                      'news_1', 'news_2', 'news_3', 'news_4_1', 'news_4_2', 'ipsos_statements_1', 'ipsos_statements_2', 
                      'ipsos_statements_3', 'ipsos_statements_4', 'ipsos_statements_5', 'ipsos_statements_6', 'ipsos_statements_7', 
                      'ipsos_statements_8'])

        q['gender_num'] = (q['gender'] == 'vrouw').astype(int)

        data_donation_query = """ SELECT u.id, u.name, AVG(
                    CASE
                        WHEN tc.topic = 'News, Politics & Societal Issues'
                        THEN 1.0
                        ELSE 0.0
                    END) AS news_ratio, AVG(tc.intolerance) AS avg_intolerance
            FROM users u
            LEFT JOIN engagements e ON e.user_id = u.id
            RIGHT JOIN transcript_classification tc ON tc.video_id = e.video_id
            GROUP BY u.id, u.name
            ORDER BY u.id ASC;
            """ 
        with self.db.connect() as db:
            data_donation = pd.read_sql(data_donation_query, db)

        return q.merge(data_donation, on='id', how='left')

    def get_cor_matrix(self, df: pd.DataFrame, columns_to_drop: list[str]=[]) -> pd.DataFrame:
        df = df.drop(columns=columns_to_drop) 
        return df.corr()

    def plot_cor_matrix(self, cor_matrix: pd.DataFrame) -> pd.DataFrame:
        plt.figure(figsize=(8, 8))
        sns.heatmap(cor_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidth=0.5)
        plt.show()


if __name__=='__main__':
    hate_target = Hate_Speech_Target()
    cor = Correlation_Matrix()

    freq = hate_target.main()
    print(freq)

    # df = cor.get_data()
    # cor_matrix = cor.get_cor_matrix(df, columns_to_drop=['id', 'cohort', 'gender', 'education', 'news_2', 'name'])
    # print(cor_matrix)
    # cor.plot_cor_matrix(cor_matrix)

