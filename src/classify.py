import pandas as pd
import sqlite3



class Classify():
    @staticmethod
    def transcipts_to_txt(self, df: str, transcriptColumn: str, descriptionColumn) -> pd.DataFrame:
        with open('transcripts.txt', 'w', encoding='utf-8') as f:
            f.write(f'{transcript}, {desc} \n' for transcript, desc in zip(df[transcriptColumn], df[descriptionColumn]))
        
        id_df: pd.DataFrame = df[['id']]
        id_df.to_csv('id.csv')
        return id_df

    @staticmethod
    def get_results(self, id_csvName: str, results_csvName: str) -> pd.DataFrame:
        results_df: pd.DataFrame = pd.read_csv(results_csvName)
        id_df: pd.DataFrame = pd.read_csv(id_csvName)

        id_df['pred'] = results_df['prob']
        id_df['prob'] = results_df['prob']

        return id_df


