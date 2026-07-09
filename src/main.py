from preprocess import *
from database import *
import pandas as pd
from classify import transcripts_to_txt, get_results

def main():
    database = Database('transcript_classification', db_path="/projects/pi-mai/Project-Data-Donation/database/db.sqlite3")
    database.enforce_correct_structure('transcript_classification') 

    """
    Classification
    """
    df = database.retrieve_data(tableName='transcript_classification tc LEFT JOIN videos v ON tc.video_id=v.id', 
                                colNames=['tc.id', 'tc.video_id', 'tc.translation', 'v.desc'], 
                                restrictions='(tc.impoliteness IS NULL) AND (tc.topic = "News, Politics & Societal Issues")', orderby='tc.id ASC', batch_size=1000)
    

    id_df = transcripts_to_txt(df, 'translation', 'desc', mergeTargetName='id')
if __name__ == '__main__':
    main()
   
