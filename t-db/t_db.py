from dsa_tdb.core import TDB_DataFrame
import dsa_tdb
import sqlite3
from pathlib import Path
from typing import Iterator
import pandas as pd
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, broadcast

BASE_DIR = Path(__file__).resolve().parent.parent

def main():
    p = Process_Tdb()
    matching_df = pd.concat(p.loop_trough_t_db())

    print(matching_df.head(10))

class Process_Tdb:
    def __init__(self, id_db_path: str=BASE_DIR/"database"/"db.sqlite3"):
        self.id_db_path: str = id_db_path
        self.spark = dsa_tdb.utils.spark_session_factory()

        self.start_date = pd.to_datetime('2023-09-01')
        self.end_date = pd.to_datetime('2026-03-01')

    def loop_trough_t_db(self, analysis_range: int=3) -> Iterator[pd.DataFrame]:
        local_start_date = self.start_date
        
        while local_start_date <= self.end_date:
        # while local_start_date <= self.start_date + pd.DateOffset(months=analysis_range):
            t_db_df: pd.DataFrame = self.get_data(start_date_ymd=local_start_date, end_date_ymd=local_start_date + pd.DateOffset(months=analysis_range))

            # matching_dfs = pd.concat(self.loop_through_id_db(t_db_df))
            id_df = self.spark.createDataFrame(self.get_ids())
            id_df = id_df.withColumn('id', col('id').cast('string'))
            matching_dfs = id_df.join(t_db_df, on='id', how='inner').toPandas()
            matching_dfs.to_csv(f'{BASE_DIR}/t-db/match_{local_start_date}.csv')
            
            local_start_date += pd.DateOffset(months=analysis_range)

            yield matching_dfs

    """
        def loop_through_id_db(self, t_db_df: pd.DataFrame, batch_size=100000) -> Iterator[pd.DataFrame]:
            offset: int = 0
            while True:
                id_df: pd.DataFrame = self.get_ids(offset=offset, batch_size=batch_size)
                if id_df.empty:
                    break

                yield self.match_ids(t_db_df, id_df) 

                offset += batch_size


        def match_ids(self, t_db_df: pd.DataFrame, id_db_df: pd.DataFrame):
            spark_ids = self.spark.createDataFrame(id_db_df)
            print(spark_ids.schema)
            spark_ids = spark_ids.withColumn('id', col('id').cast('string'))

            return broadcast(spark_ids).join(t_db_df, on='id', how='inner')
            # return spark_ids.join(t_db_df, on='id', how='inner').toPandas()
    """

    def get_data(self, start_date_ymd: str, end_date_ymd: str) -> pd.DataFrame:
        dataset_dir = BASE_DIR/"t-db/data/py_raw"
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir, ignore_errors=True)

        df = TDB_DataFrame(spark=self.spark)
        df.loadData(root_folder=dataset_dir, platform='tiktok', version='full', 
                    start_date=start_date_ymd, end_date=end_date_ymd, 
                    columns_to_import=['platform_uid', 'content_date', 'category', 'incompatible_content_ground'])

        spark_df = df._df.select(col('platform_uid').alias('id'), 'content_date', 'category', 'incompatible_content_ground')
        return spark_df

    def get_ids(self) -> pd.DataFrame:
        query = f"""SELECT id FROM videos"""
        with sqlite3.connect(self.id_db_path) as db:
           id_df = pd.read_sql(query, db)

        return id_df

    
if __name__=='__main__':
    main()
