from preprocess import *
from database import *
import pandas as pd

def main():
    preprocess = Preprocess()
    database = Database('transcript_classification')
    
    df = database.retrieve_data(tableName='transcript_classification',
                           colNames=['id, video_id, translation, filtered'],
                                restrictions="transcript LIKE '%{\"utter%' AND translation IS NOT NULL")

    print(df)

    cleaned = preprocess.preprocess_text(df, 'translation')
    print(cleaned)
    filtered = preprocess.filter_on_heuristics(df, 'translation')

    database.update_data('transcript_classification', filtered)

if __name__ == '__main__':
    main()
   
