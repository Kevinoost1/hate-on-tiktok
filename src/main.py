from preprocess import *
from database import *
import pandas as pd
from classify import transcripts_to_txt

def main():
    preprocess = Preprocess()
    database = Database('transcript_classification')
    database.enforce_correct_structure('transcript_classification') 
    
    df = database.retrieve_data(tableName='transcript_classification LEFT JOIN videos v ON tc.video_id=v.id', 
                                colNames=['tc.id', 'tc.video_id', 'tc.translation', 'v.desc'], 
                                restrictions='(topic IS NULL) AND (filtered = 0)', batch_size=10000)

    id_df = transcripts_to_txt(df, 'translation, 'desc')  

if __name__ == '__main__':
    main()
   
