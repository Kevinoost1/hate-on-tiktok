import pandas as pd
import sqlite3
from database import *
import json
import random

'''
In main.py you can find an example use for the entire analysis pipeline

*Contains all methods related to the classification*
- The two seperate methods, transcripts_to_txt() and get_results(), work together with choicellm (see readme).

- transcripts_to_txt() -> this can be called before classification by choicellm to convert your transcripts column in a pandas dataframe, to a .txt format, required by choicellm. 
    - just pass your dataframe, the name of the column with the transcripts and the name of the column with the video's description elements.
    - It will then create a 'transcripts.txt for use with choicellm and an id.csv for later merging

- get_results() -> when choicellm is finished with the classification, it will output a 'results.csv'.
    - This function merges the 'results.csv' with your 'id.csv' to ensure proper matching in the database.
      It will only take the prediction and the probabilities. If you want more data from the 'results.csv', you can add a similar line for this in the function.
'''


def transcripts_to_txt(df: pd.DataFrame, transcriptColumn: str='translation', descriptionColumn: str='desc', mergeTargetName:str='video_id') -> pd.DataFrame:
    with open('transcripts.txt', 'w', encoding='utf-8') as f:
        for t, d in zip(df[transcriptColumn], df[descriptionColumn]):
            t_clean = str(t).replace('\n', ' ').replace('\r', ' ')
            d_clean = str(d).replace('\n', ' ').replace('\r', ' ')

            f.write(t_clean + ' ' + d_clean + '\n')    

    id_df: pd.DataFrame = df[[mergeTargetName]]
    id_df.to_csv('id.csv', index=False)
    return id_df


def get_results(id_csvName: str='id.csv', results_csvName: str='results.csv', 
                predColumnName: str='pred', probColumName: str='probs', resultsColumnName: str='pred') -> pd.DataFrame:

    results_df: pd.DataFrame = pd.read_csv(results_csvName)
    id_df: pd.DataFrame = pd.read_csv(id_csvName)

    assert len(results_df) == len(id_df)
 
    id_df[predColumnName] = results_df[resultsColumnName]
    id_df[probColumName] = results_df['probs']

    return id_df


if __name__=='__main__':
    db = Database('transcript_classification', db_path="/projects/pi-mai/Project-Data-Donation/database/db.sqlite3")
    id_df = get_results(predColumnName='intolerance', probColumName='intolerance_prob', resultsColumnName='rating')

    db.update_data('transcript_classification', id_df)
    

