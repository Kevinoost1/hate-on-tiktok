import numpy as np
import pandas as pd
from database import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def main():
    db = Database('transcript_classification')
    anal = Analysis()
    data = DataExtract(db)
    df = data.request_time_series()
    print(df.head())

    plot = anal.construct_politics_t_new(df)
    plot.show()


class Analysis:
    def __init__(self):
        pass


    def construct_politics_t_new(self, timeSeriesDf: pd.DataFrame, dateColumnName: str='date_ymd', topicColumnName: str='ll3_topic_desc', cutoff: str=100) -> plt.Figure:
        df: pd.DataFrame = timeSeriesDf.copy()
        df[dateColumnName] = pd.to_datetime(timeSeriesDf[dateColumnName], yearfirst=True)        
        df['news_politics'] = (df[topicColumnName] == 'Politics & Societal Issues').astype(int)

        df: pd.DataFrame = df.set_index(dateColumnName)
        df: pd.DataFrame = df.resample('2D').agg(n_videos=('video_id', 'count'), politics_videos=('news_politics', 'sum'))
        df['ratio_news'] = df['politics_videos'] / df['n_videos']

        # only keep the rows where there is 'enough' data
        df:pd.DataFrame = df[df['n_videos'] >= cutoff]      # TO DO: make it so it only keeps the bins that have over {cutoff} support and have x bins before over {cutoff support}

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df.index, df['ratio_news'], marker='o')
        ax.set_xlabel('Starting Date')
        ax.set_ylabel('Ratio News, Politics & Societal Issues Videos')
        ax.set_title('Ratio News, Politics & Societal Issues Videos over Time')

        plt.xticks(rotation=45)
        plt.tight_layout()

        return plt
        

    def construct_politics_t(self, timeSeriesDf: pd.DataFrame, dateColumnName: str='date_ymd', topicColumnName: str='ll3_topic_desc', cutoff=50) -> plt.Figure:
        timeSeriesDf[dateColumnName] = pd.to_datetime(timeSeriesDf[dateColumnName], yearfirst=True)
         
        def construct_data(df) -> pd.DataFrame:
            df: pd.DataFrame = df.sort_values(dateColumnName)

            startDate: int = df[dateColumnName].min()
            endDate: int = df[dateColumnName].max()
            diffInterval: float = (endDate - startDate).days

            if diffInterval <= 1:
                return [df]

            n_videos: int = (df[topicColumnName] == 'Politics & Societal Issues').sum()
            df['n_videos'] = n_videos
            df['ratio_news'] = n_videos / len(df[topicColumnName])
            df['start_date'] = startDate


            middlePoint: float = startDate + pd.Timedelta(days=diffInterval / 2)
            leftSide: pd.DataFrame = df[df[dateColumnName] <= middlePoint]
            rightSide: pd.DataFrame = df[df[dateColumnName] > middlePoint]

            return construct_data(leftSide) + construct_data(rightSide)

        df: pd.DataFrame = pd.concat(construct_data(timeSeriesDf))
        df: pd.DataFrame = df[df['n_videos'] >= 50]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['start_date'], df['ratio_news'], marker='o')
        ax.set_xlabel('Starting Date')
        ax.set_ylabel('Ratio News, Politics & Societal Issues Videos')
        ax.set_title('Weekly Ratio News, Politics & Societal Issues Videos')

        plt.xticks(rotation=45)
        plt.tight_layout()

        return plt


class DataExtract:
    def __init__(self, database: Database):
        self.database: Database = database

    def request_time_series(self) -> pd.DataFrame:
        classifications: pd.DataFrame = self.database.retrieve_data(tableName='transcript_classification', 
                                                                    colNames=['video_id', 'll3_topic_desc', 'll3_sent_cat_pred', 'intolerance', 'impoliteness', 'll3_subtopic_single_pred'])
        dates: pd.DataFrame = self.database.retrieve_data(tableName='engagements',
                                                         colNames=['id', 'user_id', 'video_id', 'type', 'date_ymd', 'time_hms'])

        timeSeriesDf: pd.DataFrame = classifications.merge(dates, on='video_id', how='left')
        return self.format_time(timeSeriesDf)

    @staticmethod
    def format_time(df: pd.DataFrame, dateColumnName: str='date_ymd', timeColumnName: str='time_hms') -> pd.DataFrame: 
        df['timestamp'] = pd.to_datetime(df[dateColumnName] + ' ' + df[timeColumnName])
        df = df.sort_values(['user_id', 'timestamp'])
        df['next_ts'] = df.groupby('user_id')['timestamp'].shift(-1)
        df['watch_time_s'] = (df['next_ts'] - df['timestamp']).dt.total_seconds().fillna(0)
        
        # a new session is defined after 5 min without scrolling
        df['new_session'] = (df['watch_time_s'] >= 300).astype(int)

        # for now, drop the 'new_session == 1' rows. At those points the watch_time_s is not accurate (artificially large)
        return df[df['new_session'] == 0]



        




if __name__=='__main__':
    main()
