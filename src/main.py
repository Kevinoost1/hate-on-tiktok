from preprocess import *
from database import *
import pandas as pd

def main():
    preprocess = Preprocess()
    database = Database('transcript_classification')
    database.enforce_correct_structure('transcript_classification') 
    
    df = database.retrieve_data(tableName='transcript_classification', colNames=['id', 'language_id', 'content'], 
                                restrictions='translation IS NULL', batch_size=10000)

    translated = preprocess.generate_translation(df, 'content', 'language_id')
    filtered = preprocess.filter_on_heuristics(df, 'translation')
    
    database.update_data('transcript_classification', filtered)

if __name__ == '__main__':
    main()
   
