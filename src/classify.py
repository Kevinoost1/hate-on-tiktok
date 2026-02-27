import pandas as pd
import sqlite3



class Classify():
    @staticmethod
    def transcripts_to_txt(df: str, transcriptColumn: str, descriptionColumn) -> pd.DataFrame:
        with open('transcripts.txt', 'w', encoding='utf-8') as f:
            f.writelines(f"{transcript}, {desc}\n" for transcript, desc in zip(df[transcriptColumn], df[descriptionColumn]))        
        id_df: pd.DataFrame = df[['id']]
        id_df.to_csv('id.csv')
        return id_df

    @staticmethod
    def get_results(id_csvName: str, results_csvName: str) -> pd.DataFrame:
        results_df: pd.DataFrame = pd.read_csv(results_csvName)
        id_df: pd.DataFrame = pd.read_csv(id_csvName)

        id_df['pred'] = results_df['prob']
        id_df['prob'] = results_df['prob']

        return id_df


