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


    def construct_politics_t(self, df: pd.DataFrame, dateColumnName: str='date_ymd') -> plt.Figure:

        def construct_data(self, df) -> pd.DataFrame:
        df = df.sort_values(dateColumnName)

        startDate: int = df[dateColumnName].max()
        endDate: int = df[dateColumnName].min()
        diffInterval: float = (end - start).days()

        if diffInterval <= 1:
            return [df]

        middlePoint: float = start + (diffInterval / 2)
        leftSide: pd.DataFrame = df[df[middlePoint] <= middlePoint]
        rightSide: pd.DataFrame = df[df[middlePoint] > middlePoint]

        return construct_politics_t(leftSide, dateColumnName) + construct_politics_t(right, dateColumnName)

        




if __name__=='__main__':
    main()
