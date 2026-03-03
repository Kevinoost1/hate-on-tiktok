import numpy as np
import pandas as pd
from database import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt



def main():
    db = Database('transcript_classification')
    anal = Analysis()
    data = DataExtract(db)
    df = data.request_time_series()
    plot = anal.construct_politics_t(df)
  
    anal.get_graph_report(plot)

class Analysis:
    def __init__(self):
        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}


    def construct_politics_t(self, timeSeriesDf: pd.DataFrame, dateColumnName: str='date_ymd', topicColumnName: str='ll3_topic_single_pred', cutoff: int=15) -> plt.Figure:
        df: pd.DataFrame = timeSeriesDf.copy()
        df[dateColumnName] = pd.to_datetime(timeSeriesDf[dateColumnName], yearfirst=True)        
        df['news_politics'] = (df[topicColumnName] == 'Politics & Social Issues').astype(int)
        df['watchtime_politics'] = df['watch_time_s'] * df['news_politics']
        
        df: pd.DataFrame = df.set_index(dateColumnName)
        df: pd.DataFrame = df.resample('2D').agg(n_videos=('video_id', 'count'), n_politics_videos=('news_politics', 'sum'), watchtime_politics=('watchtime_politics', 'sum'))
        df['ratio_news'] = df['n_politics_videos'] / df['n_videos']
        df['avg_watchtime_politics'] = df['watchtime_politics'] / df['n_politics_videos']

        # only keep the rows where there is 'enough' data
        df: pd.DataFrame = df[df['n_politics_videos'] >= cutoff]      # TO DO: make it so it only keeps the bins that have over {cutoff} support and have x bins before over {cutoff support}

        political_dates: dict[str, pd.Datetime] = {
            'Dutch Election (Schoof I)' : pd.to_datetime('2023-11-22'),
            'Finalized Coalition Plans Schoof I': pd.to_datetime('2024-05-16'), 
            'U.S. Election (Trump 2025)': pd.to_datetime('2024-11-05'),
            'Inauguration and First Day Trump': pd.to_datetime('2025-01-20'),
            'Fall of Schoof I administration' : pd.to_datetime('2025-06-03'), 
            'Dutch Election (Jetten I)': pd.to_datetime('2025-10-29')
        }

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        for label, date in political_dates.items():
            for ax in axes:
                ax.axvline(date, linestyle='--', label=label, color=self.colors['green'])
        axes[0].legend()

        axes[0].plot(df.index, df['ratio_news'], marker='o', color=self.colors['dark_blue'])
        axes[1].plot(df.index, df['avg_watchtime_politics'], marker='o', color=self.colors['brown'])
        axes[1].set_xlabel('Starting Date')
        axes[0].set_ylabel('Ratio News, Politics & Societal Issues Videos')
        axes[1].set_ylabel('Mean Watch Time (sec) on News, Politics & Societal Issues Videos')
        axes[0].set_title('Ratio News, Politics & Societal Issues Videos over Time')

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        return fig

    def get_graph_report(self, *figs: tuple[plt.Figure, ...]) -> None:
        with PdfPages('graphs.pdf') as pdf:
            for fig in figs:
                pdf.savefig(fig)
                pdf.close()
            

        

class DataExtract:
    def __init__(self, database: Database):
        self.database: Database = database

    def request_time_series(self) -> pd.DataFrame:
        classifications: pd.DataFrame = self.database.retrieve_data(tableName='transcript_classification', 
                                                                    colNames=['video_id', 'll3_topic_single_pred', 'll3_sent_cat_pred', 'intolerance', 'impoliteness', 'll3_subtopic_single_pred'])
        dates: pd.DataFrame = self.database.retrieve_data(tableName='engagements',
                                                         colNames=['id', 'user_id', 'video_id', 'type', 'date_ymd', 'time_hms'])

        timeSeriesDf: pd.DataFrame = self.format_time(dates)
        return timeSeriesDf.merge(classifications, on='video_id', how='left')

    @staticmethod
    def format_time(df: pd.DataFrame, dateColumnName: str='date_ymd', timeColumnName: str='time_hms') -> pd.DataFrame: 
        df['timestamp'] = pd.to_datetime(df[dateColumnName] + ' ' + df[timeColumnName])
        df = df.sort_values(['user_id', 'timestamp'])
        df['next_ts'] = df.groupby('user_id')['timestamp'].shift(-1)
        df['watch_time_s'] = (df['next_ts'] - df['timestamp']).dt.total_seconds().fillna(0)
        
        # a new session is defined after 5 min without scrolling
        df['new_session'] = (df['watch_time_s'] >= 300).astype(int)
        df.loc[df.groupby('user_id').tail(1).index, 'new_session'] = 2
        
        # for now, drop the 'new_session == 1' rows. At those points the watch_time_s is not accurate (artificially large)
        return df[df['new_session'] == 0]



        




if __name__=='__main__':
    main()
