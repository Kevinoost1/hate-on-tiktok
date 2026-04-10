import numpy as np
import pandas as pd
from database import *
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import json
import seaborn as sns
import scipy.stats as stats
from lifelines import KaplanMeierFitter, CoxPHFitter
import zstandard as zstd
from pathlib import Path


"""
*Contains all analysis and analysis preprocessing code*
NOTE. this file contains some comments with the tag: RD; this is to indicate where a research decision is made in the code. They are either self-explanatory or explained in the paper

The file is structured in three parts:
- Fetch the appropriate dataframe from the database -> DataExtract()
- Do some basic ordering and preprocessing on that dataframe, depending on the analysis -> PrepareData()
- Fetch the Graphs -> Graphs()
- Compute the test statistics and descriptives -> Statistics()
  
DataExtract()
DataExtract will take as input a Database() Object. If you are just working with a DataFrame, you can skip this class. The code should still work if your DataFrame is similar.
The class always extracts a complete timeseries dataframe with individual feeds. Even though this is quite memory intensive, from this one dataframe you are able to construct all individual analyses, no need for multiple different SQL queries. 

    - The class also takes as arguments: get_qs, to indicate if you want to questionnaire data merged as well and channelOrdering, this is important for the multiverse analysis part but can be insanely memory intensive due to it constructing dataframes for x many definitions. 
    - A .json file (channel_def.json) is used to store complete lists of official channels under different definitions. This file should be included on the GitHub if you need it. 
    - The class has two seperate staticmethods to parse the video_duration from compressed strings. This is a very lengthy and computationally expensive process, thus it's set up to run and save seperately to later merge with the timeseries. You can run it all at once if you want (it just takes a while and needs some memory), just return the dataframe and merge it. 
    - The DataExtract() class does not simply return one dataframe, but instead a list of dataframes to work with the multiverse analysis (even if you set channelOrdering to False). 
        You can process each part of the list seperately as shown in main() and all functions still work if the length of the list is 1 (channelOrdering == False). 

PrepareData()
Not much needs to be said here. Methods within this class just take a DataFrame (timeseries) and prepare it for the downstream analyses and graphs.
There are some research decisions in there, such as a watch time cutoff and the number of days for resampling the news over time analysis. 

Graphs()
Contains the code for any of the graphs used in the paper. 
The class methods also take a (prepared) DataFrame and save all the graphs to a pdf so that it's easy to view and paste from. 

Statistics()
Methods in this class calculate statistics for any of the analyses based on the prepared dataframe.
Each analysis has its own method with a seperate descriptives method for retrieving the sample descriptives.
All results are saved to a .json so it's easy to copy from and view at later moments without rerunning analyses. 
"""


BASE_DIR = Path(__file__).resolve().parent.parent


def main():
    db = Database('transcript_classification', db_path=BASE_DIR / 'database' / 'old_db.sqlite3')
    prep = PrepareData()
    data = DataExtract(db)
    graphs = Graphs()
    stats = Statistics()

    # dfs: list[pd.DataFrame] | list[pd.DataFrame, ...] = data.request_time_series(get_qs=True, channelOrdering=True)
    
    # politics_t_dfs: list[pd.DataFrame] = [prep.construct_politics_t(df) for df in dfs]
    # civility_dfs: list[pd.DataFrame] | list[pd.DataFrame, ...] = [prep.construct_civility(df) for df in dfs]
    # survival_dfs: list[pd.DataFrame] | list[pd.DataFrame, ...] = [prep.construct_survival(df) for df in dfs]   

    # desc = stats.get_descriptives(dfs)
    # stats.get_civility(civility_dfs)
    # stats.get_survival(survival_dfs)
    # stats.dump_to_json()

    # politics_t_plot: plt.Figure = graphs.create_timeseries_graph(politics_t_dfs) 
    # civility_plot: plt.Figure = graphs.create_civility_graph(civility_dfs)
    # km_plot: plt.Figure = graphs.create_km_graph(survival_dfs)
 
    # graphs.get_graph_report(politics_t_plot, civility_plot, km_plot)

    data.extract_duration()


class PrepareData():
    def __init__(self):
        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}


    def construct_politics_t(self, timeSeriesDf: pd.DataFrame, dateColumnName: str='date_ymd', topicColumnName: str='ll3_topic_single_pred', cutoff: int=8) -> pd.DataFrame:
        print('it reached construct politics')
        df: pd.DataFrame = timeSeriesDf.copy()
        df[dateColumnName] = pd.to_datetime(timeSeriesDf[dateColumnName], yearfirst=True) 
        df['news_politics'] = (df[topicColumnName] == 'Politics & Social Issues').astype(int)
        df['watchtime_politics'] = df['watch_time_s'] * df['news_politics']
        
        df: pd.DataFrame = df.set_index(dateColumnName)
        df['high_intolerance'] = (df['intolerance'] > 5).astype(int)

        user_dfs = []
        for user, user_df in df.groupby('user_id'):
            resampled: pd.DataFrame = user_df.resample('5D').agg(n_videos=('video_id', 'count'), n_politics_videos=('news_politics', 'sum'), watchtime_politics=('watchtime_politics', 'sum'), 
                                                                 n_high_intolerance=('high_intolerance', 'sum'))
            resampled['ratio_news'] = resampled['n_politics_videos'] / resampled['n_videos']
            resampled['avg_watchtime_politics'] = resampled['watchtime_politics'] / resampled['n_politics_videos']
            resampled['ratio_high_intolerance'] = resampled['n_high_intolerance'] / resampled['n_politics_videos']
            user_dfs.append(resampled)

        df: pd.DataFrame = pd.concat(user_dfs)
        # RD: only keep the rows where there is 'enough' data
        df: pd.DataFrame = df[df['n_politics_videos'] >= cutoff]      # TO DO: make it so it only keeps the bins that have over {cutoff} support and have x bins before over {cutoff support}

        df:pd.DataFrame = df.groupby(df.index).agg(mean_ratio=('ratio_news', 'mean'), sem_ratio=('ratio_news', lambda x: x.std()/np.sqrt(len(x))), 
                                                   mean_watch_time=('avg_watchtime_politics', 'mean'), sem_watch_time=('avg_watchtime_politics', lambda x: x.std()/np.sqrt(len(x))),
                                                   mean_intolerance=('ratio_high_intolerance', 'mean'), sem_intolerance=('ratio_high_intolerance', lambda x: x.std()/np.sqrt(len(x))))

        df['mean_ratio'] = df['mean_ratio'].rolling(window=5, min_periods=1, center=True).mean()
        df['mean_watch_time'] = df['mean_watch_time'].rolling(window=5, min_periods=1, center=True).mean()
        df['mean_intolerance'] = df['mean_intolerance'].rolling(window=5, min_periods=1, center=True).mean()

        return df


    def construct_civility(self, timeSeriesDf: pd.DataFrame, topicColumnName:str='ll3_topic_single_pred', intoleranceColumnName: str='intolerance', impolitenessColumnName: str='impoliteness') -> pd.DataFrame:
        print('it reached civility')
        df: pd.DataFrame = timeSeriesDf.copy()[['user_id', 'official', topicColumnName, intoleranceColumnName, impolitenessColumnName]]
        df: pd.DataFrame = df[df[topicColumnName] == 'Politics & Social Issues']
 
        results: pd.DataFrame = (df.groupby(['user_id', 'official'], as_index=False).agg(
            avg_impoliteness=(impolitenessColumnName, 'mean'),
            avg_intolerance=(intoleranceColumnName, 'mean')
        ))
        return results

    def construct_survival(self, timeSeriesDf: pd.DataFrame, topicColumnName:str= 'll3_topic_single_pred', sentColumnName: str='ll3_sent_cat_pred', cutoff: int=3) -> pd.DataFrame:
        df: pd.DataFrame = timeSeriesDf.copy()[[topicColumnName, sentColumnName, 'watch_time_s', 'age']].assign(event=1)

        # RD: A cutoff was set at 3 sec, in line with Meta's policy for Instagram, to count a video as viewed
        df = df[df['watch_time_s'] >= cutoff]

        df.loc[df['ll3_topic_single_pred'] != 'Politics & Social Issues', 'll3_topic_single_pred'] = 'Other'
        return df
        
class Graphs():
    def __init__(self):
        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}


    # TO DO: make the graph nicer    
    def create_km_graph(self, dfs: list[pd.DataFrame], topicColumnName: str='ll3_topic_single_pred', sentColumnName: str='ll3_sent_cat_pred') -> plt.Figure():
        df: pd.DataFrame = dfs[0]

        km = KaplanMeierFitter()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for i, (classification, column) in enumerate({'topic': topicColumnName, 'sent': sentColumnName}.items()):
            ax_classification = axes[i]

            for idx, (label, df_label) in enumerate(df.groupby(column)):
                km.fit(durations=df_label['watch_time_s'], event_observed=df_label['event'], label=str(label))
                km.plot_survival_function(ax=ax_classification, color=list(self.colors.values())[idx])

            ax_classification.set_title(f"Surivival by {classification}")
            ax_classification.set_xlabel('Watch Time (sec)')
            ax_classification.set_ylabel('Proportion of Video\'s Still Watched')

        axes[0].set_xlim(3, 250)
        axes[0].set_title("Watch Time on TikTok Video's as Seperated by Topic")

        axes[1].set_xlim(3, 250)
        axes[1].set_title("Watch Time on TikTok Video's as Seperated by Sentiment")

        plt.tight_layout()

        return fig

    # TO DO: make the graph nicer
    def create_timeseries_graph(self, dfs: list[pd.DataFrame]): 
        df = dfs[0]

        political_dates: dict[str, pd.Datetime] = {
            'Dutch Election (Schoof I)' : pd.to_datetime('2023-11-22'),
            'Finalized Coalition Plans Schoof I': pd.to_datetime('2024-05-16'), 
            'U.S. Election (Trump 2025)': pd.to_datetime('2024-11-05'),
            'Inauguration and First Day Trump': pd.to_datetime('2025-01-20'),
            'Fall of Schoof I administration' : pd.to_datetime('2025-06-03'), 
            'Dutch Election (Jetten I)': pd.to_datetime('2025-10-29')
        }

        fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)

        for label, date in political_dates.items():
            for ax in axes:
                ax.axvline(date, linestyle='--', label=label, color=self.colors['green'])
        axes[0].legend()

        axes[0].plot(df.index, df['mean_ratio'], marker='o', color=self.colors['dark_blue'])
        axes[0].fill_between(df.index, df['mean_ratio'] - df['sem_ratio'], df['mean_ratio'] + df['sem_ratio'], color=self.colors['dark_blue'], alpha=0.3)
        axes[0].set_ylabel('Ratio News, Politics & Societal Issues Videos')
        axes[0].set_ylim(0, (df['mean_ratio'] + df['sem_ratio']).max())
        axes[0].set_title('Ratio News, Politics & Societal Issues Videos over Time')
                
        axes[1].plot(df.index, df['mean_watch_time'], marker='o', color=self.colors['brown'])
        axes[1].fill_between(df.index, df['mean_watch_time'] - df['sem_watch_time'], df['mean_watch_time'] + df['sem_watch_time'], color=self.colors['brown'], alpha=0.3)
        axes[1].set_xlabel('Starting Date')
        axes[1].set_ylim(0, (df['mean_watch_time'] + df['sem_watch_time']).max())
        axes[1].set_ylabel('Mean Watch Time (sec) on News, Politics & Societal Issues Videos')

        axes[2].plot(df.index, df['mean_intolerance'], marker='o', color=self.colors['purple'])
        axes[2].fill_between(df.index, df['mean_intolerance'] - df['sem_intolerance'], df['mean_intolerance'] + df['sem_intolerance'], color=self.colors['purple'], alpha=0.3)
        axes[2].set_ylabel('Mean Intolerance Rating')
        axes[2].set_ylim(0, (df['mean_intolerance'] + df['sem_intolerance']).max())

        plt.xticks(rotation=45)
        plt.tight_layout()

        return fig       

    def create_civility_graph(self, dfs: list[pd.DataFrame]):
        results = dfs
        fig, axes = plt.subplots(2, 1, figsize=(8,6), sharex=True)

        for df in results:
            sns.pointplot(data=df, x="official", y="avg_impoliteness", estimator="mean", errorbar="se", join=True, markers="o", color=self.colors['orange'], ax=axes[0])
            sns.pointplot(data=df, x="official", y="avg_intolerance", estimator="mean", errorbar="se", join=True, markers="o", color=self.colors['orange'], ax=axes[1])
        
        combined = pd.concat(results, ignore_index=True)
        avg_over_definitions = combined.groupby(['user_id', 'official'], as_index=False).agg(avg_intolerance=('avg_intolerance', 'mean'), avg_impoliteness=('avg_impoliteness', 'mean'))

        sns.pointplot(data=avg_over_definitions, x="official", y="avg_impoliteness", estimator="mean", errorbar="se", join=True, markers="o", color=self.colors['dark_blue'], ax=axes[0])
        sns.stripplot(data=avg_over_definitions, x="official", y="avg_impoliteness", jitter=0.2, alpha=0.2, ax=axes[0], size=3, color=self.colors['dark_blue'])

        sns.pointplot(data=avg_over_definitions, x="official", y="avg_intolerance", estimator="mean", errorbar="se", join=True, markers="o", color=self.colors['dark_blue'], ax=axes[1])
        sns.stripplot(data=avg_over_definitions, x="official", y="avg_intolerance", jitter=0.2, alpha=0.2, ax=axes[1], size=3, color=self.colors['dark_blue'])


        axes[0].set_ylabel("Average Impoliteness")
        axes[0].set_title("Average Civility per Channel Type")
        axes[0].set_yticks(np.arange(avg_over_definitions["avg_impoliteness"].min(), avg_over_definitions["avg_impoliteness"].max(), 0.5))

        axes[1].set_ylabel("Average Intolerance")
        axes[1].set_xticklabels(["Unofficial","Official"])
        axes[1].set_xlabel("")
        axes[1].set_yticks(np.arange(avg_over_definitions["avg_intolerance"].min(), avg_over_definitions["avg_intolerance"].max(), 0.5))

        return fig

    def get_graph_report(self, *figs: tuple[plt.Figure, ...]) -> None:
        with PdfPages('graphs.pdf') as pdf:
            for fig in figs:
                pdf.savefig(fig) 

class Statistics:
    def __init__(self, outputName: str='stats.json'):
        self.outputName: str = outputName
        self.jsonContent: dict[str, list[str, float]] = {'Descriptives': [], 'Watchtime':[], 'Civility':[], 'Timeseries': []}
        
    def get_descriptives(self, timeSeriesDfs) -> dict[str, dict[str, float]]:
        df: pd.DataFrame = timeSeriesDfs[0].copy()

        df['news_politics'] = (df['ll3_topic_single_pred'] == 'Politics & Social Issues').astype(int)
        df['videos_with_transcript'] = (df['ll3_topic_single_pred'].notna()).astype(int)
        for sent in ['Positive', 'Negative', 'Neutral']:
            df[sent] = (df['ll3_sent_cat_pred'] == sent).astype(int)

        per_user_num: pd.DataFrame = df.groupby('user_id').agg(
            avg_intolerance=('intolerance', 'mean'),
            avg_impoliteness=('impoliteness', 'mean'),
            n_videos=('video_id', 'count'),
            n_unique_videos=('video_id', 'nunique'),
            n_videos_w_transcript=('videos_with_transcript', 'sum'),
            n_unique_videos_w_transcript=('video_id',  lambda x: x[df.loc[x.index, 'videos_with_transcript'] == 1].nunique()),
            n_politics=('news_politics', 'sum'),
            Positive=('Positive', 'sum'), 
            Negative=('Negative', 'sum'), 
            Neutral=('Neutral', 'sum'), 

            age=('age', 'last'),
            political_preference=('political_preference', 'first'),

        )   
        per_user_num['ratio_news'] = per_user_num['n_politics'] / per_user_num['n_videos']
        for sent in ['Positive', 'Negative', 'Neutral']:
            per_user_num[f'ratio_{sent}'] = per_user_num[sent] / per_user_num['n_videos']

        per_user_cat: pd.DataFrame = df.groupby('user_id').agg(gender=('gender', 'first'), education=('education', 'first'))
 
        descriptives_num = per_user_num.agg(['mean', 'std', 'median', 'max', 'min', 'count', 'sum']).to_dict()
        descriptives_cat = {col: per_user_cat[col].value_counts().to_dict() for col in per_user_cat.columns}

        descriptives = {'Numeric': descriptives_num, 'Categorical': descriptives_cat}
        
        self.jsonContent['Descriptives'].append(descriptives)

        return descriptives

    def get_civility(self, civilityDfs) -> dict[str, float]:
        dfs: pd.DataFrame = civilityDfs

        p_values = {'intolerance': [], 'impoliteness': []}
        t_values = {'intolerance': [], 'impoliteness': []}
        d_values = {'intolerance': [], 'impoliteness': []}

        results = {'p_values': p_values, 't_values': t_values, 'd_values': d_values, 'd_mean': {}, 'd_sd': {}, 'd_range': {}, 'p_range': {}}

        for df in dfs:
            paired: pd.DataFrame = df.pivot(index='user_id', columns='official', values=['avg_intolerance', 'avg_impoliteness'])

            # RD: Some people had NaN (likely to little videos from 'official')       # TO DO: remember what was up with these people
            paired = paired.dropna()

            int_results = stats.ttest_rel(paired['avg_intolerance'][1], paired['avg_intolerance'][0], alternative='less') 
            imp_results = stats.ttest_rel(paired['avg_impoliteness'][1], paired['avg_impoliteness'][0], alternative='less')

            p_values['intolerance'].append(int_results.pvalue)
            p_values['impoliteness'].append(imp_results.pvalue)

            d_values['intolerance'].append(int_results.statistic / np.sqrt(len(paired['avg_intolerance'][1])))
            d_values['impoliteness'].append(imp_results.statistic / np.sqrt(len(paired['avg_impoliteness'][1])))

            t_values['intolerance'].append(int_results.statistic)
            t_values['impoliteness'].append(imp_results.statistic)

        for analysis_type in ['intolerance', 'impoliteness']:
            d_array = np.array(d_values[analysis_type])
            p_array = np.array(p_values[analysis_type])

            results['d_mean'][analysis_type] = np.mean(d_array)
            results['d_sd'][analysis_type] = np.std(d_array)
            results['d_range'][analysis_type] = (np.min(d_array), np.max(d_array))
            results['p_range'][analysis_type] = (np.min(p_array), np.max(p_array))

        self.jsonContent['Civility'].append(results)

        return results

    def get_survival(self, survival_dfs: pd.DataFrame):
        df = survival_dfs[0][['ll3_topic_single_pred', 'll3_sent_cat_pred', 'event', 'watch_time_s']].copy()

        # RD: Using Other and Neutral as references
        df['ll3_topic_single_pred'] = pd.Categorical(df['ll3_topic_single_pred'], categories=['Other', 'Politics & Social Issues'])
        df['ll3_sent_cat_pred'] = pd.Categorical(df['ll3_sent_cat_pred'], categories=['Neutral', 'Negative', 'Positive'])
        df = pd.get_dummies(df, columns=['ll3_topic_single_pred', 'll3_sent_cat_pred'], drop_first=True)

        cph = CoxPHFitter()
        cph.fit(df, duration_col='watch_time_s', event_col='event')

        summary = cph.summary.to_dict()
        self.jsonContent['Watchtime'].append(summary)

        return summary
        
    def dump_to_json(self):
        with open('stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.jsonContent, f, indent=4)

 

class DataExtract:
    def __init__(self, database: Database, base_dir=BASE_DIR):
        self.database: Database = database
        self.base_dir = base_dir

    def request_time_series(self, channelOrdering: bool=False, get_qs: bool=False) -> list[pd.DataFrame] | list[pd.DataFrame, ...]:
        classifications: pd.DataFrame = self.database.retrieve_data(tableName='transcript_classification', 
                                                                    colNames=['video_id', 'll3_topic_single_pred', 'translation', 'll3_sent_cat_pred', 'intolerance', 'impoliteness', 'll3_subtopic_single_pred'])
        dates: pd.DataFrame = self.database.retrieve_data(tableName='engagements',
                                                         colNames=['id', 'user_id', 'video_id', 'type', 'date_ymd', 'time_hms'])

        videos = self.database.retrieve_data(tableName='videos', colNames=['id AS video_id, author_id'])

        official, unofficial, ordering = self.load_channels() 
        channels: pd.DataFrame = self.database.retrieve_data(tableName='authors', colNames=['id AS author_id', 'unique_id'])
        channels['official'] = channels['unique_id'].apply(lambda x: (x in official)).astype(int)

        timeSeriesDf: pd.DataFrame = self.format_time(dates)
        timeSeriesDf: pd.DataFrame = timeSeriesDf.merge(classifications, on='video_id', how='left') \
            .merge(videos, on='video_id', how='left') \
            .merge(channels, on='author_id', how='left')

        if get_qs:
            qs: pd.DataFrame = self.database.retrieve_data(tableName='questionnaires', colNames=['name', 'age', 'gender', 'education', 'political_preference'])
            users: pd.DataFrame = self.database.retrieve_data(tableName='users', colNames=['id AS user_id', 'name']).merge(qs, on='name', how='inner')
            timeSeriesDf: pd.DataFrame = timeSeriesDf.merge(users, on='user_id', how='inner').drop(columns=['name']) 
            timeSeriesDf = self.rollback_age(timeSeriesDf)
        
        timeSeriesDfs: list[pd.DataFrame, ...] = [timeSeriesDf.copy()]
        if channelOrdering: 
            for n in range(len(ordering)):
                mask = timeSeriesDf['unique_id'].isin(ordering[str(n)])
                timeSeriesDf.loc[timeSeriesDf['unique_id'].isin(ordering[str(n)]), 'official'] = 0
                
                timeSeriesDfs.append(timeSeriesDf.copy())

        return timeSeriesDfs
        
    @staticmethod
    def format_time(df: pd.DataFrame, dateColumnName: str='date_ymd', timeColumnName: str='time_hms') -> pd.DataFrame: 
        df['timestamp'] = pd.to_datetime(df[dateColumnName] + ' ' + df[timeColumnName])
        df = df.sort_values(['user_id', 'timestamp'])
        df['next_ts'] = df.groupby('user_id')['timestamp'].shift(-1)
        df['watch_time_s'] = (df['next_ts'] - df['timestamp']).dt.total_seconds().fillna(0)
        
        # RD: a new session is defined after 5 min without scrolling
        df['new_session'] = (df['watch_time_s'] >= 300).astype(int)
        df.loc[df.groupby('user_id').tail(1).index, 'new_session'] = 2
        
        # RD: drop the 'new_session != 0' rows. At those points the watch_time_s is not accurate (artificially large)
        return df[df['new_session'] == 0]

    # TO DO: implement retrieve_video_durations()
    #   - Find a way to make it memory efficient or to clear memory when it is full

    @staticmethod
    def rollback_age(df: pd.DataFrame, dateColumnName: str='date_ymd') -> pd.DataFrame:
        df[dateColumnName] = pd.to_datetime(df[dateColumnName], yearfirst=True)
        last_date = df.groupby('user_id')[dateColumnName].transform('max')

        delta_year = last_date.dt.year - df[dateColumnName].dt.year
        df['age'] = df['age'] - delta_year
        return df

    @staticmethod
    def load_channels(channelDefinitionFileName: str=BASE_DIR / 'src' / 'channel_def.json') -> (list[str, ...], list[str, ...], list[str, ...]):
        with open(channelDefinitionFileName, 'r') as f:
            channels = json.load(f)
        
        return channels['official'], channels['unofficial'], channels['ordering']

    def extract_duration(self):
        duration_df = self.database.retrieve_data(tableName='transcript_classification tc JOIN videos_raw v ON v.id=tc.video_id', 
                                                  colNames=['v.id AS video_id', 'v.video_data_zstd'])['video_data_zstd'].apply(self.decode_json)
        duration_df.to_csv('duration.csv')
    
    @staticmethod
    def decode_json(compressed_string):
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed_string)
        raw_json = json.loads(decompressed.decode("utf-8")) 
        try:
            if "video" in raw_json and "duration" in raw_json["video"]:
                duration = raw_json["video"]["duration"]
            else:
                duration = None

        except Exception:
            duration = None

        return duration




if __name__=='__main__':
    main()
