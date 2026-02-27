import pandas as pd
import sqlite3


'''
In main.py you can find an example use for the entire analysis pipeline

*Contains all methods related to the classification*
- The two seperate methods, transcripts_to_txt() and get_results(), work together with choicellm (see readme).

- transcripts_to_txt() -> this can be called before classification by choicellm to convert your transcripts column in a pandas dataframe, to a .txt format, required by choicellm. 
    - just pass your dataframe, the name of the column with the transcripts and the name of the column with the video's description elements.
    - It will then create a 'transcripts.txt for use with choicellm and an id.csv for later merging

- get_results() -> when choicellm is finished with the classification, it will output a 'results.csv'.
    - This function merges the 'results.csv' with your 'id.csv' to ensure proper matching in the database.
      It will only take the prediction
'''


def transcripts_to_txt(df: str, transcriptColumn: str, descriptionColumn) -> pd.DataFrame:
    with open('transcripts.txt', 'w', encoding='utf-8') as f:
        f.writelines(f"{transcript}, {desc}\n" for transcript, desc in zip(df[transcriptColumn], df[descriptionColumn]))        
    id_df: pd.DataFrame = df[['id']]
    id_df.to_csv('id.csv')
    return id_df

def get_results(id_csvName: str, results_csvName: str) -> pd.DataFrame:
    results_df: pd.DataFrame = pd.read_csv(results_csvName)
    id_df: pd.DataFrame = pd.read_csv(id_csvName)

    id_df['pred'] = results_df['pred']
    id_df['prob'] = results_df['prob']

    return id_df


