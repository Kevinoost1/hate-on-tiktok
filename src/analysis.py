import numpy as np
import pandas as pd
from database import *
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def main():
    pass


class Analysis:
    def __init__(self):
        pass


    def construct_politics_t(self, timeSeriesDf: pd.DataFrame, dateColumnName: str='date_ymd', topicColumnName: str='ll3_topic_desc') -> plt.Figure:
        df[dateColumnName] = pd.to_datetime(df[dateColumnName], yearfirst=True)
         
        def construct_data(self, df) -> pd.DataFrame:
            df: pd.DataFrame = df.sort_values(dateColumnName)

            startDate: int = df[dateColumnName].min()
            endDate: int = df[dateColumnName].max()
            diffInterval: float = (end - start).days()

            if diffInterval <= 1:
                return [df]

            n_videos: int = sum(df[df[topicColumnName] == 'Politics & Societal Issues'].astype(int))
            df['n_videos'] = n_videos
            df['ratio_news']: pd.Series = n_videos / len(df[topicColumnName])
            df['start_date']: pd.Series = df[startDate]


            middlePoint: float = start + (diffInterval / 2)
            leftSide: pd.DataFrame = df[df[middlePoint] <= middlePoint]
            rightSide: pd.DataFrame = df[df[middlePoint] > middlePoint]

            return construct_politics_t(leftSide, dateColumnName) + construct_politics_t(right, dateColumnName)

        df: pd.DataFrame = pd.concat(construct_data(df))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df['start_date'], df['ratio_news'], marker='o')
        ax.set_xlabel('Starting Date')
        ax.set_ylabel('Ratio News, Politics & Societal Issues Videos')
        ax.set_title('Weekly Ratio News, Politics & Societal Issues Videos')

        plt.xticks(rotation=45)
        plt.tight_layout()

        return plt

        




if __name__=='__main__':
    main()
