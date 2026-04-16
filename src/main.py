from preprocess import *
from database import *
import pandas as pd
from classify import transcripts_to_txt, get_results

def main():
    preprocess = Preprocess()
    database = Database('transcript_classification')
    database.enforce_correct_structure('transcript_classification') 


    df = database.retrieve_data(tableName='transcript_classification tc LEFT JOIN videos v ON tc.video_id=v.id', 
                                colNames=['tc.id', 'tc.video_id', 'tc.translation', 'v.desc'], 
                                restrictions='(tc.topic IS NULL) AND (tc.filtered = 0)', orderby='tc.id ASC', batch_size=10)
    

    id_df = transcripts_to_txt(df, 'translation', 'desc', mergeTargetName='id')
if __name__ == '__main__':
    main()
   
