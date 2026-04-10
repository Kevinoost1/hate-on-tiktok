import pandas as pd
import sqlite3
from database import *
import json

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


def transcripts_to_txt(df: pd.DataFrame, transcriptColumn: str, descriptionColumn, mergeTargetName:str='video_id') -> pd.DataFrame:
    with open('transcripts.txt', 'w', encoding='utf-8') as f:
        for t, d in zip(df[transcriptColumn], df[descriptionColumn]):
            f.write(json.dumps({"transcript": t, "desc": d}) + "\n")

    id_df: pd.DataFrame = df[[mergeTargetName]]
    id_df.to_csv('id.csv', index=False)
    return id_df

def fake():
    preds = []
    probs = []

    with open('transcripts.txt', 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            preds.append(f"ROW_{i}")
            probs.append(i)

    return pd.DataFrame({
    'pred': preds,
    'probs': probs
    })      

def get_results(id_csvName: str='id.csv', results_csvName: str='results.csv', 
                predColumnName: str='pred', probColumName: str='probs') -> pd.DataFrame:
    # results_df: pd.DataFrame = pd.read_csv(results_csvName)
    results_df = fake()
    id_df: pd.DataFrame = pd.read_csv(id_csvName)

    assert len(results_df) == len(id_df)

    print(len(id_df))
    print(len(results_df))

    id_df[predColumnName] = results_df['pred']
    id_df[probColumName] = results_df['probs']

    return id_df


if __name__=='__main__':
    db = Database('transcript_classification')
    id_df = get_results()
    db.update_data('transcript_classification', id_df, predColumnName='topic', probColumName='topic_prob')
    

