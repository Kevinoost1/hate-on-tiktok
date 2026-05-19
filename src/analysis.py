import numpy as np
import pandas as pd
from database import *
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import seaborn as sns
import scipy.stats as stats
from lifelines import KaplanMeierFitter, CoxPHFitter
import zstandard as zstd
from pathlib import Path
from mne.stats import permutation_cluster_1samp_test
from statsmodels.stats.multitest import multipletests
import math
import matplotlib.dates as mdates



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
    - The class has two seperate staticmethods to parse the video_duration from compressed strings. It just has to be run once for the video durations to be decompressed and added to the database
    - The DataExtract() class does not simply return one dataframe, but instead a list of dataframes to work with the multiverse analysis (even if you set channelOrdering to False). 
        You can process each part of the list seperately as shown in main() and all functions still work if the length of the list is 1 (channelOrdering == False). 

NOTE. I won't go into every specific analysis here and why it's conducted. You can find that information in the paper. 

PrepareData()
Not much needs to be said here. Methods within this class just take a DataFrame (timeseries) and prepare it for the downstream analyses and graphs.
There are some research decisions in there, such as a watch time cutoff and the number of days for resampling the news over time analysis. 

Graphs()
Contains the code for any of the graphs used in the paper. 
The class methods also take a (prepared) DataFrame and save all the graphs to a pdf so that it's easy to view and paste from and as more high quality .pngs into a folder /plots.

Statistics()
Methods in this class calculate statistics for any of the analyses based on the prepared dataframe.
Each analysis has its own method with a seperate descriptives method for retrieving the sample descriptives.
All results are saved to a .json so it's easy to copy from and view at later moments without rerunning analyses. 
"""


BASE_DIR = Path(__file__).resolve().parent.parent
TOPIC = 'News, Politics & Societal Issues'


def main():
    db = Database(db_path=BASE_DIR / 'database' / 'db.sqlite3')
    prep = PrepareData()
    data = DataExtract(db)
    graphs = Graphs()
    stats = Statistics()

    dfs: list[pd.DataFrame] = data.request_time_series(get_qs=True, channelOrdering=True)
    
    """
    Preprocess for each analysis
    """
    politics_t_dfs: list[pd.DataFrame] = [prep.preprocess_epochs(df) for df in dfs]
    politics_t_event_dfs = [df[0] for df in politics_t_dfs]
    politics_t_baseline_dfs = [df[1] for df in politics_t_dfs]

    civility_dfs: list[pd.DataFrame] = [prep.construct_civility(df) for df in dfs]
    survival_dfs: list[pd.DataFrame] = [prep.construct_survival(df) for df in dfs]   

    """
    Statistics for each analysis and descriptives
    """
    desc = stats.get_descriptives(dfs)
    stats.get_civility(civility_dfs)
    stats.get_survival(survival_dfs)
    stats.get_timeseries(politics_t_event_dfs[0], politics_t_baseline_dfs[0])
    stats.dump_to_json()

    """
    Graphs for each analysis
    """
    politics_t_plot: plt.Figure = graphs.create_timeseries_graph(dfs) 
    civility_plot: plt.Figure = graphs.create_civility_graph(civility_dfs)
    km_plot: plt.Figure = graphs.create_km_graph(survival_dfs) 

    graphs.get_graph_report(politics_t_plot, civility_plot, km_plot)
    graphs.export_png(politics_t_plot, civility_plot, km_plot)


class PrepareData():
    def __init__(self, topicColumnName='topic', intoleranceColumnName='intolerance'):
        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}
        self.political_events = {
            'Dutch Election (Schoof I)' : pd.to_datetime('2023-11-22'),
            'Finalized Coalition Plans Schoof I': pd.to_datetime('2024-05-16'), 
            'U.S. Election (Trump 2025)': pd.to_datetime('2024-11-05'),
            'Inauguration and First Day Trump': pd.to_datetime('2025-01-20'),
            'Fall of Schoof I administration' : pd.to_datetime('2025-06-03'), 
            'Dutch Election (Jetten I)': pd.to_datetime('2025-10-29')
        }

        self.topicColumnName: str = topicColumnName
        self.intoleranceColumnName: str = intoleranceColumnName

    """
    Preprocessing for cluster permutation t-test 
    -> create analysis windows as epochs.
    -> for each analysis window compute a baseline window to correct with.
    """
    def preprocess_epochs(self, timeSeriesDf: pd.DataFrame, pre_window=14, post_window=3, non_event_window=28) -> (pd.DataFrame, pd.DataFrame):
        df: pd.DataFrame = timeSeriesDf.copy()
        df['date_ymd'] = pd.to_datetime(df['date_ymd'])

        epochs: list[None] | list[pd.DataFrame] = []
        baselines: list[None] | list[pd.DataFrame] = []

        for name, event in self.political_events.items():
            relative_day = (df['date_ymd'] - event).dt.days
            in_window = relative_day.between(-pre_window, post_window)

            temp: pd.DataFrame = df.loc[(in_window)].copy()
            temp['event'] = name
            temp['relative_day'] = relative_day[in_window]
            epochs.append(temp)

            in_baseline_window = relative_day.between(-non_event_window, -pre_window - 1)
            temp_base: pd.DataFrame = df.loc[in_baseline_window].copy()
            temp_base['event'] = name
            temp_base['relative_day'] = relative_day[in_baseline_window]
            baselines.append(temp_base)

        df_epochs = pd.concat(epochs, ignore_index=True)
        df_baseline = pd.concat(baselines, ignore_index=True)

        for d in [df_epochs, df_baseline]:
            d['is_news'] = (d[self.topicColumnName] == TOPIC).astype(int)
            d['high_int'] = (d[self.intoleranceColumnName] >= 5).astype(int)

        event_df = df_epochs.groupby(['user_id', 'event', 'relative_day']).agg(n_news=('is_news', 'sum'), n_total=('is_news', 'count'), n_civility=('high_int', 'sum')).reset_index()
        event_df['ratio'] = event_df['n_news'] / event_df['n_total']
        event_df['int_ratio'] = event_df['n_civility'] / event_df['n_news']

        baseline = df_baseline.groupby(['user_id', 'event']).agg(n_news=('is_news', 'sum'), n_total=('is_news', 'count'), n_civility=('high_int', 'sum')).reset_index()
        baseline['baseline_ratio'] = baseline['n_news'] / baseline['n_total']
        baseline['baseline_int_ratio'] = baseline['n_civility'] / baseline['n_news']

        return event_df, baseline

    """
    Preprocessing for channel-level civility independent t-test
    -> Only take the channels that are preselected (official == 1 or official == 2)
    -> Group per channel and take the mean
    NOTE. This method works with the multiverse analysis
    """
    def construct_civility(self, timeSeriesDf: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = timeSeriesDf.copy()[['user_id', 'unique_id', 'official', self.topicColumnName, self.intoleranceColumnName]] #, 'impoliteness']]
 
        df = df[(df[self.topicColumnName] == TOPIC) & (df['official'] != 0)]
        
        results: pd.DataFrame = (df.groupby(['unique_id', 'official'], as_index=False). agg(
            # avg_impoliteness=('impoliteness', 'mean'),
            avg_intolerance=(self.intoleranceColumnName, 'mean')
        ))
        results['official'] = results['official'].astype(int)
        return results

    """
    Preprocessing for survival analysis
    -> Create an event variable (always one after t time since every video is eventually scrolled by
    -> Only focus on news vs. not news
    """
    def construct_survival(self, timeSeriesDf: pd.DataFrame, sentColumnName: str='sent', cutoff: int=3) -> pd.DataFrame:
        df: pd.DataFrame = timeSeriesDf.copy()[[self.topicColumnName, sentColumnName, 'watch_time_s', 'age', 'video_duration']].assign(event=1)

        # RD: A cutoff was set at 3 sec, in line with Meta's policy for Instagram, to count a video as viewed
        df = df[df['watch_time_s'] >= cutoff]

        df.loc[df[self.topicColumnName] != TOPIC, self.topicColumnName] = 'Other'
        return df
        
class Graphs():
    def __init__(self, topicColumnName='topic', intoleranceColumnName='intolerance'):
        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}
        self.political_events = {
            'Election (NL)' : pd.to_datetime('2023-11-22'),
            'Coalition Plans (NL)': pd.to_datetime('2024-05-16'), 
            'Election (U.S.)': pd.to_datetime('2024-11-05'),
            'Presidential Inauguration': pd.to_datetime('2025-01-20'),
            'Government Fall (NL)' : pd.to_datetime('2025-06-03'), 
            'Election (NL)': pd.to_datetime('2025-10-29')
        }

        self.topicColumnName: str = topicColumnName
        self.intoleranceColumnName: str = intoleranceColumnName

        self.fontsize_labels: int = 30
        self.fontsize_titles: int = 36
        self.axesticks: int = 28

    # TO DO: make the graph nicer    
    def create_km_graph(self, dfs: list[pd.DataFrame], sentColumnName: str='sent') -> plt.Figure:
        df: pd.DataFrame = dfs[0]

        km = KaplanMeierFitter()
        fig, axes = plt.subplots(1, 2, figsize=(40, 20))

        for i, (classification, column) in enumerate({'topic': self.topicColumnName, 'sent': sentColumnName}.items()):
            ax_classification = axes[i]

            for idx, (label, df_label) in enumerate(df.groupby(column)):
                km.fit(durations=df_label['watch_time_s'], event_observed=df_label['event'], label=str(label))
                km.plot_survival_function(ax=ax_classification, color=list(self.colors.values())[idx])

            ax_classification.legend(fontsize=30)

            ax_classification.set_title(f"Surivival by {classification}", fontsize=self.fontsize_titles)
            ax_classification.set_xlabel('Watch Time (sec)', fontsize=self.fontsize_labels)
            ax_classification.set_ylabel('Proportion of Video\'s Still Watched', fontsize=self.fontsize_labels)

        axes[0].set_xlim(3, 250)
        axes[0].set_title("Watch Time on TikTok Video's as Seperated by Topic", fontsize=self.fontsize_titles)
        axes[0].tick_params(axis='both', labelsize=self.axesticks)
 
        axes[1].set_xlim(3, 250)
        axes[1].set_title("Watch Time on TikTok Video's as Seperated by Sentiment", fontsize=self.fontsize_titles)
        axes[1].tick_params(axis='both', labelsize=self.axesticks)

        return fig


    # TO DO: make the graph nicer
    def create_timeseries_graph(self, timeSeriesDfs) -> plt.Figure: 
        df: pd.DataFrame = timeSeriesDfs[0].copy()

        df['is_news'] = (df[self.topicColumnName] == TOPIC).astype(int)
        df['high_int'] = (df[self.intoleranceColumnName] >= 5).astype(int)

        per_user: pd.DataFrame = df.groupby(['user_id', 'date_ymd']).agg(n_news=('is_news', 'sum'), n_total=('is_news', 'count'), n_high_int=('high_int', 'sum')).reset_index()
        per_user['ratio'] = per_user['n_news'] / per_user['n_total']
        per_user['int_ratio'] = per_user['n_high_int'] / per_user['n_news']

        grand_avg: pd.DataFrame = per_user.groupby('date_ymd').agg(mean_ratio=('ratio', 'mean'), sem_ratio=('ratio', lambda x: x.std() / np.sqrt(len(x))), 
                                                    n_users=('user_id', 'nunique'), n_videos=('n_total', 'sum'), 
                                                     mean_int_ratio=('int_ratio', 'mean'), sem_int_ratio=('int_ratio', lambda x: x.std() / np.sqrt(len(x)))).reset_index()
 
        transp = (grand_avg['n_users'] - grand_avg['n_users'].min()) / (grand_avg['n_users'].max() - grand_avg['n_users'].min())
        transp = transp.fillna(0.1)

        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(40, 20), gridspec_kw={'height_ratios': [2, 2, 1, 1]})
        for i, row in grand_avg.iterrows():
            axes[0].scatter(row['date_ymd'], row['mean_ratio'], color=self.colors['dark_blue'], alpha=transp.iloc[i], s=15)
            axes[1].scatter(row['date_ymd'], row['mean_int_ratio'], color=self.colors['purple'], alpha=transp.iloc[i], s=15)

            assert not (row['n_videos'] > 0 and row['n_users'] == 0)


        axes[0].plot(grand_avg['date_ymd'], grand_avg['mean_ratio'], color=self.colors['dark_blue'], linewidth=2.5)
        axes[0].fill_between(grand_avg['date_ymd'], grand_avg['mean_ratio'] - grand_avg['sem_ratio'], grand_avg['mean_ratio'] + grand_avg['sem_ratio'], color=self.colors['dark_blue'], alpha=0.15)
        axes[1].plot(grand_avg['date_ymd'], grand_avg['mean_int_ratio'], color=self.colors['purple'], linewidth=2.5)
        axes[1].fill_between(grand_avg['date_ymd'], grand_avg['mean_int_ratio'] - grand_avg['sem_int_ratio'], grand_avg['mean_int_ratio'] + grand_avg['sem_int_ratio'], color=self.colors['purple'], alpha=0.15)

        legend = []
        for label, date in self.political_events.items():
            for i, plot in enumerate(axes):
                axes[i].axvline(date, linestyle='--', color=self.colors['green'], alpha=0.8)
                axes[i].axvspan(date - pd.Timedelta(days=14), date + pd.Timedelta(days=3), color=self.colors['green'], alpha=0.08)

            # legend.append(Line2D([0], [0], color=self.colors['green'], linestyle='--', label=label))
            axes[3].text(date, -0.1, label, rotation=45, ha='right', va='top', fontsize=self.axesticks, color=self.colors['green'], transform=axes[3].get_xaxis_transform())

        axes[0].set_ylabel('Ratio News in Feed', fontsize=27)
        axes[0].set_title("Exposure to News, Politics & Societal Issues Content and Intolerance over Time", fontsize=self.fontsize_titles)
        # axes[0].legend(handles=legend, fontsize=18, frameon=False, loc='upper left', bbox_to_anchor=(-2, 1.55))
        axes[0].set_xlim(pd.to_datetime('2023-10-01'), pd.to_datetime('2026-01-01'))
        axes[0].set_ylim(0, 0.3)

        axes[1].set_ylabel('Ratio Intolerant\nNews in Feed', fontsize=self.fontsize_labels)
        axes[1].set_ylim(0, 0.3)

        axes[2].plot(grand_avg['date_ymd'], grand_avg['n_videos'], color=self.colors['brown'], linewidth=1.5, label='Number of Videos')
        axes[2].set_ylabel('Number of\nVideos', fontsize=self.fontsize_labels)

        axes[3].plot(grand_avg['date_ymd'], grand_avg['n_users'], color=self.colors['orange'], alpha=0.5, label='Number of Participants')
        axes[3].set_ylabel('Number of\nParticipants', fontsize=self.fontsize_labels)

        for ax in axes:
            ax.tick_params(axis='both', labelsize=self.axesticks)

        ticks = pd.date_range(start='2023-10-01', end='2026-01-01', freq='9MS')
        axes[3].set_xticks(ticks)
        axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.xticks(rotation=45)
        fig.subplots_adjust(bottom=0.22)

        return fig

    def create_civility_graph(self, dfs: list[pd.DataFrame]) -> plt.Figure:
        results: list[pd.DataFrame] = dfs
        fig, axes = plt.subplots(1, 1, figsize=(20,20), sharex=True)

        for i, df in enumerate(results):
            t = i / max(len(results) - 1, 1)
            alpha = 0.2 + (1 - 0.2) * (1 - t)**5

            # sns.pointplot(data=df, x="official", y="avg_impoliteness", estimator="mean", errorbar="se", join=True, markers="o", color=self.colors['orange'], ax=axes[0])
            sns.pointplot(data=df, x="official", y="avg_intolerance", estimator="mean", errorbar="se", join=True, markers="o", color=self.colors['purple'], ax=axes, linewidth=1, alpha=alpha)
        
        # combined: pd.DataFrame = pd.concat(results, ignore_index=True)
        # avg_over_definitions: pd.DataFrame = combined.groupby(['unique_id', 'official'], as_index=False).agg(avg_intolerance=('avg_intolerance', 'mean')) # , avg_impoliteness=('avg_impoliteness', 'mean'))

        # sns.pointplot(data=avg_over_definitions, x="official", y="avg_impoliteness", estimator="mean", errorbar="se", join=True, markers="o", color=self.colors['dark_blue'], ax=axes[0])
        # sns.pointplot(data=avg_over_definitions, x="official", y="avg_intolerance", estimator="mean", errorbar="se", join=True, markers="o", color=self.colors['dark_blue'], ax=axes[1])

        # axes[0].set_ylabel("Average Impoliteness")
        axes.set_title("Average Civility per Channel Type", fontsize=self.fontsize_titles)
        # axes[0].set_yticks(np.arange(avg_over_definitions["avg_impoliteness"].min(), avg_over_definitions["avg_impoliteness"].max(), 0.5))

        axes.set_ylabel("Average Intolerance", fontsize=self.fontsize_labels)
        axes.set_xticklabels(["Official", "Unofficial"])
        axes.tick_params(axis='both', labelsize=self.axesticks)
        axes.set_xlabel("")
        # axes[1].set_yticks(np.arange(avg_over_definitions["avg_intolerance"].min(), avg_over_definitions["avg_intolerance"].max(), 0.5))

        return fig


    def get_graph_report(self, *figs: tuple[plt.Figure]) -> None:
        with PdfPages('graphs.pdf') as pdf:
            for fig in figs:
                pdf.savefig(fig) 

    def export_png(self,  *figs: tuple[plt.Figure]) -> None:
        for i, fig in enumerate(figs):
            fig.savefig(f'plots/{i}.png', dpi=300)

class Statistics:
    def __init__(self, outputName: str='stats.json', topicColumnName='topic'):
        self.outputName: str = outputName
        self.jsonContent: dict[str, list[str, float]] = {'Descriptives': [], 'Watchtime':[], 'Civility':[], 'Timeseries': []}
        self.political_events = {
            'Dutch Election (Schoof I)' : pd.to_datetime('2023-11-22'),
            'Finalized Coalition Plans Schoof I': pd.to_datetime('2024-05-16'), 
            'U.S. Election (Trump 2025)': pd.to_datetime('2024-11-05'),
            'Inauguration and First Day Trump': pd.to_datetime('2025-01-20'),
            'Fall of Schoof I administration' : pd.to_datetime('2025-06-03'), 
            'Dutch Election (Jetten I)': pd.to_datetime('2025-10-29')
        }

        self.topicColumnName: str = topicColumnName
        
    def get_descriptives(self, timeSeriesDfs) -> dict[str, dict[str, float]]:
        df: pd.DataFrame = timeSeriesDfs[0].copy()
        df['date_ymd'] = pd.to_datetime(df['date_ymd'])

        df['news_politics'] = (df[self.topicColumnName] == TOPIC).astype(int)
        df['videos_with_transcript'] = (df[self.topicColumnName].notna()).astype(int)
        for sent in ['Positive', 'Negative', 'Neutral']:
            df[sent] = (df['sent'] == sent).astype(int)

        per_user_num: pd.DataFrame = df.groupby('user_id').agg(
            avg_intolerance=('intolerance', 'mean'),
            # avg_impoliteness=('impoliteness', 'mean'),
            n_videos=('video_id', 'count'),
            dates=('date_ymd', lambda x: (x.max() - x.min()).days),
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

    """
    Cluster permutation t-test
    -> Bonferoni-Holm correction at then end
    """ 
    def get_timeseries(self, politics_t_event_df, politics_t_baseline_df): 
        results = {}

        df = politics_t_event_df.merge(politics_t_baseline_df, on=['user_id', 'event'], how='inner')
        df['c_news_ratio'] = df['ratio'] - df['baseline_ratio']
        df['c_int_ratio'] = df['int_ratio'] - df['baseline_int_ratio']
        p_val = {'news': [], 'int': []}

        event_order = df['event'].unique()

        for event in event_order:
            local_results = {}

            for analysis_type, col in {'news': 'c_news_ratio', 'int': 'c_int_ratio'}.items():
                df_event = df[df['event'] == event].pivot(index='user_id', columns='relative_day', values=col).sort_index(axis=1)

                # RD: participants who were active in the analysis window, but not on a specific day get 0 on their ratios after baseline correction, essentially pre-analysis window mean imputation
                df_event = df_event.fillna(0)
                        
                x = df_event.values
                t, clusters, p, _ = permutation_cluster_1samp_test(x, tail=1, n_permutations=1000, out_type='mask')

                p_val[analysis_type].append(np.min(p) if len(p) > 0 else 1)
                
                timepoints = df_event.columns.values
                local_results[analysis_type] = {'t': t.tolist(), 
                                                'clusters': [{'start_day': int(timepoints[c[0].start]), 'end_day': int(timepoints[c[0].stop - 1])} for c in clusters], 
                                                'p': p.tolist()}
            results[event] = local_results

        for analysis_type in ['news', 'int']:
            nope, p_cor, _, _ = multipletests(p_val[analysis_type], method='holm')

            for i, event in enumerate(event_order):
                results[event][analysis_type]['significant'] = bool(nope[i])
                results[event][analysis_type]['p_corrected'] = float(p_cor[i])

        self.jsonContent['Timeseries'].append(results)
        return results


    """
    Independent t-test for the channel-level civility
    NOTE. this works with the multiverse analysis
    """
    def get_civility(self, civilityDfs) -> dict[str, float]:
        dfs: pd.DataFrame = civilityDfs

        p_values = {'intolerance': [], 'impoliteness': []}
        t_values = {'intolerance': [], 'impoliteness': []}
        d_values = {'intolerance': [], 'impoliteness': []}

        results = {'p_values': p_values, 't_values': t_values, 'd_values': d_values, 'd_mean': {}, 'd_sd': {}, 'd_range': {}, 'p_range': {}}

        for df in dfs:
            int_results = stats.ttest_ind(df.loc[df['official'] == 1, 'avg_intolerance'], df.loc[df['official'] == 2, 'avg_intolerance'], alternative='less') 
            # imp_results = stats.ttest_ind(df.loc[df['official'] == 1, 'avg_impoliteness'], df.loc[df['official'] == 2, 'avg_impoliteness'], alternative='less') 

            p_values['intolerance'].append(int_results.pvalue)
            # p_values['impoliteness'].append(imp_results.pvalue)

            def cohens_d(official, unofficial):
                n_official, n_unofficial = len(official), len(unofficial)
                var_official, var_unofficial = official.var(), unofficial.var()

                pooled_sd = np.sqrt(((n_official - 1) * var_official + (n_unofficial - 1) * var_unofficial) / (n_official + n_unofficial - 2))
                return (official.mean() - unofficial.mean()) / pooled_sd

            d_values['intolerance'].append(cohens_d(df.loc[df['official'] == 1, 'avg_intolerance'], df.loc[df['official'] == 2, 'avg_intolerance']))
            # d_values['impoliteness'].append(cohens_d(df.loc[df['official'] == 1, 'avg_impoliteness'], df.loc[df['official'] == 2, 'avg_impoliteness']))

            t_values['intolerance'].append(int_results.statistic)
            # t_values['impoliteness'].append(imp_results.statistic)

        for analysis_type in ['intolerance']: # , 'impoliteness']:
            d_array = np.array(d_values[analysis_type])
            p_array = np.array(p_values[analysis_type])

            results['d_mean'][analysis_type] = np.mean(d_array)
            results['d_sd'][analysis_type] = np.std(d_array)
            results['d_range'][analysis_type] = (np.min(d_array), np.max(d_array))
            results['p_range'][analysis_type] = (np.min(p_array), np.max(p_array))

        self.jsonContent['Civility'].append(results)

        return results

    """
    Cox Regression
    """
    def get_survival(self, survival_dfs: pd.DataFrame):
        df = survival_dfs[0][[self.topicColumnName, 'sent', 'event', 'watch_time_s', 'video_duration']].copy()

        df[self.topicColumnName] = pd.Categorical(df[self.topicColumnName], categories=['Other', TOPIC])
        df['sent'] = pd.Categorical(df['sent'], categories=['Neutral', 'Negative', 'Positive'])
        df = pd.get_dummies(df, columns=[self.topicColumnName, 'sent'], drop_first=False)

        # RD: Using Other and Neutral as references
        df = df.drop(columns=[f'{self.topicColumnName}_Other','sent_Neutral'])

        cph = CoxPHFitter()
        cph.fit(df, duration_col='watch_time_s', event_col='event')

        summary = cph.summary.to_dict()
        self.jsonContent['Watchtime'].append(summary)

        return summary   
            
    def dump_to_json(self):
        with open('stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.jsonContent, f, indent=4)

 

class DataExtract:
    def __init__(self, database: Database, base_dir=BASE_DIR, topicColumnName='topic', 
                 sentColumnName='sent', intoleranceColumnName='intolerance', # impolitenessColumnName='impoliteness', 
                 subtopicColumnName='subtopic'):
        self.database: Database = database
        self.base_dir = base_dir

        self.topicColumnName: str = topicColumnName
        self.sentColumnName: str = sentColumnName
        self.intoleranceColumnName: str = intoleranceColumnName
        # self.impolitenessColumnName: str = impolitenessColumnName
        self.subtopicColumnName: str = subtopicColumnName

    def request_time_series(self, channelOrdering: bool=False, get_qs: bool=False) -> list[pd.DataFrame] | list[pd.DataFrame, ...]:
        # Add impolitness?
        classifications: pd.DataFrame = self.database.retrieve_data(tableName='transcript_classification', 
                                                                    colNames=['video_id', self.topicColumnName, 'translation', self.sentColumnName, self.intoleranceColumnName, self.subtopicColumnName, 'video_duration'])
        dates: pd.DataFrame = self.database.retrieve_data(tableName='engagements',
                                                         colNames=['id', 'user_id', 'video_id', 'type', 'date_ymd', 'time_hms'], restrictions='user_id < 2')

        videos: pd.DataFrame = self.database.retrieve_data(tableName='videos', colNames=['id AS video_id, author_id'])

        official, unofficial, ordering = self.load_channels() 

        channels: pd.DataFrame = self.database.retrieve_data(tableName='authors', colNames=['id AS author_id', 'unique_id'])
        channels['official'] = np.select([channels['unique_id'].isin(official), channels['unique_id'].isin(unofficial)], [1, 2], default=0)

        timeSeriesDf: pd.DataFrame = self.format_time(dates)
        timeSeriesDf: pd.DataFrame = timeSeriesDf.merge(classifications, on='video_id', how='left') \
            .merge(videos, on='video_id', how='left') \
            .merge(channels, on='author_id', how='left')

        if get_qs:
            qs: pd.DataFrame = self.database.retrieve_data(tableName='questionnaires', colNames=['name', 'age', 'gender', 'education', 'political_preference'])
            users: pd.DataFrame = self.database.retrieve_data(tableName='users', colNames=['id AS user_id', 'name']).merge(qs, on='name', how='inner')
            timeSeriesDf: pd.DataFrame = timeSeriesDf.merge(users, on='user_id', how='inner').drop(columns=['name']) 
            timeSeriesDf = self.rollback_age(timeSeriesDf)
        
        # RD: we only take the part that has classifications. We don't use the other video's for now
        timeSeriesDf = timeSeriesDf[timeSeriesDf['topic'].notna()].copy()

        timeSeriesDfs: list[pd.DataFrame, ...] = [timeSeriesDf.copy()]

        if channelOrdering: 
            for n in range(len(ordering)):
                def_df = timeSeriesDf.copy() 
                def_df.loc[def_df['unique_id'].isin(ordering[str(n)]), 'official'] = 2 
                timeSeriesDfs.append(def_df)

                # assert def_df.loc[def_df['unique_id'] == 'foxnews', 'official'].iloc[0] == (1 if n < 5 else 2)

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

    @staticmethod
    def rollback_age(df: pd.DataFrame, dateColumnName: str='date_ymd') -> pd.DataFrame:
        df[dateColumnName] = pd.to_datetime(df[dateColumnName], yearfirst=True)
        last_date = df.groupby('user_id')[dateColumnName].transform('max')

        delta_year = last_date.dt.year - df[dateColumnName].dt.year
        df['age'] = df['age'] - delta_year
        return df

    @staticmethod
    def load_channels(channelDefinitionFileName: str=BASE_DIR / 'src' / 'channel_def.json') -> (list[str], list[str], list[str]):
        with open(channelDefinitionFileName, 'r') as f:
            channels = json.load(f)
        
        return channels['official'], channels['unofficial'], channels['ordering']


    # When you run this, convert row values to int in the Database() class
    def extract_duration(self):
        batch_size=1000
        while True:
            duration_df = self.database.retrieve_data(tableName='transcript_classification tc INNER JOIN videos_raw v ON v.id=tc.video_id', 
                                                    colNames=['tc.id', 'v.video_data_zstd AS video_duration'], restrictions='tc.video_duration IS NULL',
                                                    batch_size=batch_size, orderby='tc.id ASC') 
            if duration_df.empty:
                break

            duration_df['video_duration'] = duration_df['video_duration'].apply(self.decode_json)
            
            self.database.update_data(tableName='transcript_classification', data=duration_df)
             
    @staticmethod
    def decode_json(compressed_string):
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed_string)
        raw_json = json.loads(decompressed.decode("utf-8")) 

        if "video" in raw_json and "duration" in raw_json["video"]:
            duration = raw_json["video"]["duration"]
        else:
            duration = None 

        return int(duration)

if __name__=='__main__':
    main()
