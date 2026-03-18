from preprocess import *
from database import *
import pandas as pd

def main():
    preprocess = Preprocess()
    database = Database('transcript_classification')
    
    df = database.retrieve_data(tableName='transcript_classification', colNames=['id, video_id, content'])
    cleaned = preprocess.preprocess_text(df, 'content')
    database.update_data('transcript_classification', cleaned)

if __name__ == '__main__':
    main()
   
