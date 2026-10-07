import pandas as pd
import sqlite3
from database import *
import subprocess
import json
import numpy as np
from mne.stats import permutation_cluster_1samp_test
from statsmodels.stats.multitest import multipletests
from lifelines import KaplanMeierFitter, CoxPHFitter 

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.use('Agg')
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
TOPIC = 'News, Politics & Societal Issues'


def main():
    db = Database(db_path=BASE_DIR / 'database' / '150_db.sqlite3')
    prep = Preprocess()
    data = DataExtract(db)
    graphs = Graphs()
    stats = Statistics()
    robust = Robustness(db)

    dfs: list[pd.DataFrame] = data.request_time_series(get_qs=False, channel_ordering=True)
    hate_diff_threshold = graphs.create_hate_hist(dfs[0])

    politics_t_dfs: list[pd.DataFrame] = [prep.preprocess_epochs(df) for df in dfs]
    politics_t_event_dfs = [df[0] for df in politics_t_dfs]
    politics_t_baseline_dfs = [df[1] for df in politics_t_dfs]

    politics_t_plot, data_availability = graphs.create_timeseries_graph(dfs) 
    hate_source_plot = graphs.create_hate_source_graph(dfs)

    graphs.get_graph_report(politics_t_plot, hate_source_plot, data_availability, hate_diff_threshold)
    graphs.export_png(politics_t_plot, hate_source_plot, data_availability, hate_diff_threshold)

    stats.get_timeseries(politics_t_event_dfs[0], politics_t_baseline_dfs[0])
    stats.get_hate_source(dfs)
    stats.get_watchtime(dfs)
    stats.dump_to_json()

    robust.robustness_hate_source(dfs)


class Preprocess:
    def __init__(self, topicColumnName='topic', intoleranceColumnName='intolerance', impolitenessColumnName='impoliteness'):
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
        self.impolitenessColumnName: str = impolitenessColumnName


    """
    Preprocessing for cluster permutation t-test 
    -> create analysis windows as epochs.
    -> for each analysis window compute a baseline window to correct with.
    """
    def preprocess_epochs(self, timeSeriesDf: pd.DataFrame, pre_window=14, post_window=3, non_event_window=28) -> (pd.DataFrame, pd.DataFrame):
        df: pd.DataFrame = timeSeriesDf.copy()

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

        event_df = df_epochs.groupby(['user_id', 'event', 'relative_day']).agg(n_news=('is_news', 'sum'), n_total=('is_news', 'count'), n_hate=('high_hate', 'sum')).reset_index()
        event_df['ratio'] = event_df['n_news'] / event_df['n_total']
        event_df['hate_ratio'] = event_df['n_hate'] / event_df['n_news']

        baseline = df_baseline.groupby(['user_id', 'event']).agg(n_news=('is_news', 'sum'), n_total=('is_news', 'count'), n_hate=('high_hate', 'sum')).reset_index()
        baseline['baseline_ratio'] = baseline['n_news'] / baseline['n_total']
        baseline['baseline_hate_ratio'] = baseline['n_hate'] / baseline['n_news']

        return event_df, baseline


class Statistics:
    def __init__(self, topicColumnName='topic', intoleranceColumnName='intolerance', impolitenessColumnName='impoliteness', sentColumnName='sent', subtopicColumnName='subtopic', 
                 outputName='stats.json'):
        self.topicColumnName: str = topicColumnName
        self.sentColumnName: str = sentColumnName
        self.intoleranceColumnName: str = intoleranceColumnName
        self.impolitenessColumnName: str = impolitenessColumnName
        self.subtopicColumnName: str = subtopicColumnName

        self.outputName: str = outputName
        self.jsonContent: dict[str, list[str, float]] = {'Descriptives': [], 'Watchtime':[], 'Hate Source':[], 'Timeseries': []}
        self.political_events = {
            'Dutch Election (Schoof I)' : pd.to_datetime('2023-11-22'),
            'Finalized Coalition Plans Schoof I': pd.to_datetime('2024-05-16'), 
            'U.S. Election (Trump 2025)': pd.to_datetime('2024-11-05'),
            'Inauguration and First Day Trump': pd.to_datetime('2025-01-20'),
            'Fall of Schoof I administration' : pd.to_datetime('2025-06-03'), 
            'Dutch Election (Jetten I)': pd.to_datetime('2025-10-29')
        }

    def get_watchtime(self, dfs):
        df = dfs[0]

        # We can only keep the videos where we can cleanly measure watch time
        df[df['new_session'] == 0]

        df.to_csv(BASE_DIR/'survival.csv')

        subprocess.run(["Rscript", str(BASE_DIR/ "src" / "survival.R")], check=True)
        subprocess.run(["rm", str(BASE_DIR/ "survival.csv")], check=True)

        results = pd.read_csv('survival_results.csv').to_dict()
        self.jsonContent['Watchtime'].append(results)
        return results
 
    def get_hate_source(self, dfs):
        results = {}

        for i, df in enumerate(dfs):
            df = df[df[self.topicColumnName] == 'News, Politics & Societal Issues']
            df.to_csv(BASE_DIR/'hate_source.csv')

            subprocess.run(["Rscript", str(BASE_DIR/ "src" / "hate_source.R")], check=True)
            subprocess.run(["rm", str(BASE_DIR/ "hate_source.csv")], check=True)

            local_results = pd.read_csv('hate_source_results.csv').to_dict()
            results[i] = local_results

        def get_results(result_type, category, mapping_dict={'intercept': 0,'legacy': 1, 'alternative': 2}, inverted=False):
            if not inverted:
                values = [r[result_type][mapping_dict[category]] for r in results.values()]
            else:
                values = [1 / r[result_type][mapping_dict[category]] for r in results.values()]

            return min(values), max(values)

        def get_results_ci(category, mapping_dict={'intercept': 0, 'legacy': 1, 'alternative': 2}):
            conf_lows = [r['conf_low_OR'][mapping_dict[category]] for r in results.values()]
            conf_highs = [r['conf_high_OR'][mapping_dict[category]]for r in results.values()]

            inverse_ci_lows = [1 / x for x in conf_highs]
            inverse_ci_highs = [1 / x for x in conf_lows]

            return min(inverse_ci_lows), max(inverse_ci_highs)            

        # NOTE. we take the inverse OR just for stylistic reasons. In the paper the conclusions are formulated as:
        # niche higher than ..., however the model gives output: ... lower odds than ... Just a matter of where u put the reference in the text
        results['global'] = {
            'intercept_min_p': get_results('p.value', 'intercept')[0], 'intercept_max_p': get_results('p.value', 'intercept')[1],
            'official1_min_p': get_results('p.value', 'legacy')[0], 'official1_max_p': get_results('p.value', 'legacy')[1],
            'official2_min_p': get_results('p.value', 'alternative')[0], 'official2_max_p': get_results('p.value', 'alternative')[1],

            'official1_min_OR_inv': get_results('odds_ratio', 'legacy', inverted=True)[0], 'official1_max_OR_inv': get_results('odds_ratio', 'legacy', inverted=True)[1],
            'official2_min_OR_inv': get_results('odds_ratio', 'alternative', inverted=True)[0], 'official2_max_OR_inv': get_results('odds_ratio', 'alternative', inverted=True)[1],           
            'intercept_min_OR_baseChance': (get_results('odds_ratio', 'intercept')[0]) / (1 + get_results('odds_ratio', 'intercept')[0]), 
                'intercept_max_OR_baseChance': ((get_results('odds_ratio', 'intercept')[1]) / (1 + get_results('odds_ratio', 'intercept')[1])),

            'intercept_min_conflow_OR_inv': get_results_ci('intercept')[0], 'intercept_max_conf_OR_inv': get_results_ci('intercept')[1],
            'official1_min_conflow_OR_inv': get_results_ci('legacy')[0], 'official1_max_conf_OR_inv': get_results_ci('legacy')[1],
            'official2_min_conflow_OR_inv': get_results_ci('alternative')[0], 'official2_max_conf_OR_inv': get_results_ci('alternative')[1],
            }

        self.jsonContent['Hate Source'].append(results)
        return results

    
    def get_timeseries(self, politics_t_event_df, politics_t_baseline_df): 
            results = {}

            df = politics_t_event_df.merge(politics_t_baseline_df, on=['user_id', 'event'], how='inner')
            df['c_news_ratio'] = df['ratio'] - df['baseline_ratio']
            df['c_hate_ratio'] = df['hate_ratio'] - df['baseline_hate_ratio']
            p_val = {'news': [], 'hate': []}

            event_order = df['event'].unique()

            for event in event_order:
                local_results = {}

                for analysis_type, col in {'news': 'c_news_ratio', 'hate': 'c_hate_ratio'}.items():
                    df_event = df[df['event'] == event].pivot(index='user_id', columns='relative_day', values=col).sort_index(axis=1)

                    # RD: participants who were active in the analysis window, but not on a specific day get 0 on their ratios after baseline correction, essentially pre-analysis window mean imputation
                    df_event = df_event.fillna(0) 
                    x = df_event.values

                    mean_ts = np.mean(x, axis=0)

                    t, clusters, p, _ = permutation_cluster_1samp_test(x, tail=1, n_permutations=1000, out_type='mask')
                    p_val[analysis_type].append(np.min(p) if len(p) > 0 else 1) 
                    timepoints = df_event.columns.values

                    # This is just for keeping the effect size uncorrected for baseline such that it is interpretable alongside the graph
                    raw_col = 'ratio' if analysis_type == 'news' else 'hate_ratio'
                    df_event = df[df['event'] == event].pivot(index='user_id', columns='relative_day', values=raw_col).sort_index(axis=1)
                    mean_ts_raw = np.nanmean(df_event.values, axis=0)

                    local_results[analysis_type] = {'t': t.tolist(), 
                                                    'clusters': [{'start_day': int(timepoints[c[0].start]), 
                                                                'end_day': int(timepoints[c[0].stop - 1]), 
                                                                'max_peak_height': float(np.max(mean_ts_raw[c[0]])), 
                                                                'auc': float(np.trapezoid(mean_ts[c[0]], x=timepoints[c[0]]))} for c in clusters],  
                                                    'p': p.tolist()}
                results[event] = local_results

            for analysis_type in ['news', 'hate']:
                nope, p_cor, _, _ = multipletests(p_val[analysis_type], method='holm')

                for i, event in enumerate(event_order):
                    results[event][analysis_type]['significant'] = bool(nope[i])
                    results[event][analysis_type]['p_corrected'] = float(p_cor[i])

            self.jsonContent['Timeseries'].append(results)
            return results

    def dump_to_json(self, label=''):
        print('this is the json content', self.jsonContent)
        with open(f'{label}stats.json', 'w', encoding='utf-8') as f:
            json.dump(self.jsonContent, f, indent=4)

class Graphs:
    def __init__(self, topicColumnName='topic', sentColumnName='sent', intoleranceColumnName='intolerance', impolitenessColumnName='impoliteness', subtopicColumnName='subtopic'):
        self.topicColumnName: str = topicColumnName
        self.sentColumnName: str = sentColumnName
        self.intoleranceColumnName: str = intoleranceColumnName
        self.impolitenessColumnName: str = impolitenessColumnName
        self.subtopicColumnName: str = subtopicColumnName

        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}
        self.fontsize_labels: int = 30
        self.fontsize_titles: int = 36
        self.axesticks: int = 26

        self.political_events = {
            '2023 Election (NL)' : pd.to_datetime('2023-11-22'),
            'Coalition Plans (NL)': pd.to_datetime('2024-05-16'), 
            '2024 Election (U.S)': pd.to_datetime('2024-11-05'),
            'Presidential Inauguration': pd.to_datetime('2025-01-20'),
            'Government Fall (NL)' : pd.to_datetime('2025-06-03'), 
            '2025 Election (NL)': pd.to_datetime('2025-10-29')
        }
                
    def create_hate_hist(self, df) -> plt.Figure:
        df = df[df[self.topicColumnName] == TOPIC].copy()

        thresholds = [2, 3, 4, 5, 6]
        proportions = []

        for threshold in thresholds:
            high_hate = ((df[self.intoleranceColumnName] >= threshold) | (df[self.impolitenessColumnName] >= threshold))
            user_proportions = (high_hate.groupby(df['user_id']).mean())

            print(f'hate ratio std at threshold {threshold}', high_hate.groupby(df['user_id']).std())
            print(f'hate ratio max at threshold {threshold}', high_hate.groupby(df['user_id']).max())
            print(f'hate ratio min at threshold {threshold}', high_hate.groupby(df['user_id']).min())

            proportions.append(user_proportions.mean())

        fig, ax = plt.subplots(figsize=(40, 40))
        ax.plot(thresholds, proportions, marker='o', linewidth=2, color=self.colors['purple'])

        for threshold, proportion in zip(thresholds, proportions):
            ax.annotate(f'{proportion:.1%}', (threshold, proportion), xytext=(8, 5), textcoords='offset points', fontsize=self.axesticks)

        ax.set_title('Average Hate Proportion By Chosen Threshold', fontsize=self.fontsize_titles)
        ax.set_xlabel('Chosen Hate Threshold', fontsize=self.fontsize_labels)
        ax.set_ylabel('Average Proportion of\nContent Classified Labeled as Hate', fontsize=self.fontsize_labels)

        ax.set_xticks(thresholds)
        ax.tick_params(axis='both', labelsize=self.axesticks)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

        ax.grid(alpha=0.3)

        return fig

    def create_hate_source_graph(self, dfs: list[pd.DataFrame]) -> plt.Figure:
        preprocessed_dfs = [] 

        for df in dfs:
            df = df[df['high_hate'] == 1].copy()
                
            results: pd.DataFrame = (df.groupby(['user_id', 'official'], as_index=False). agg(
                n_hate=('high_hate', 'sum')
            ))

            results['official'] = results['official'].astype(int)
            results['n_total'] = (results.groupby('user_id')['n_hate'].transform('sum'))

            results['ratio_hate'] = (results['n_hate'] /results['n_total'])

            plot_results = results[
                results['official'].isin([1, 2])
            ].copy()

            preprocessed_dfs.append(plot_results)



        results: list[pd.DataFrame] = preprocessed_dfs 
        fig, axes = plt.subplots(1, 1, figsize=(20,20), sharex=True)

        for i, df in enumerate(results):
            t = i / max(len(results) - 1, 1)
            alpha = 0.2 + (1 - 0.2) * t**5

            sns.pointplot(data=df, x="official", y="ratio_hate", estimator="mean", errorbar="se", join=True, markers="o", color=self.colors['purple'], ax=axes, linewidth=1, alpha=alpha)
        
        axes.set_title("Average Ratio of Total Hate for Legacy and Alternative\nAcross Multiverse Definitions\n", fontsize=self.fontsize_titles)

        axes.set_ylabel("Average Ratio of Total Hate", fontsize=self.fontsize_labels)
        axes.set_xticklabels(["Legacy Channels", "Alternative Channels", "Niche Channels"])
        axes.tick_params(axis='both', labelsize=self.axesticks)
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)

        axes.set_xlabel("")
        
        plt.show()

        return fig        

    def create_timeseries_graph(self, timeSeriesDfs) -> plt.Figure: 
        df: pd.DataFrame = timeSeriesDfs[0].copy()

        per_user: pd.DataFrame = df.groupby(['user_id', 'date_ymd']).agg(n_news=('is_news', 'sum'), n_total=('is_news', 'count'), n_high_hate=('high_hate', 'sum')).reset_index()
        per_user['ratio'] = per_user['n_news'] / per_user['n_total']
        per_user['hate_ratio'] = per_user['n_high_hate'] / per_user['n_news']

        grand_avg: pd.DataFrame = per_user.groupby('date_ymd').agg(mean_ratio=('ratio', 'mean'), sem_ratio=('ratio', lambda x: x.std() / np.sqrt(len(x))), 
                                                    n_users=('user_id', 'nunique'), n_videos=('n_total', 'sum'), 
                                                     mean_hate_ratio=('hate_ratio', 'mean'), sem_hate_ratio=('hate_ratio', lambda x: x.std() / np.sqrt(len(x)))).reset_index()
 
        transp = (grand_avg['n_users'] - grand_avg['n_users'].min()) / (grand_avg['n_users'].max() - grand_avg['n_users'].min())
        transp = transp.fillna(0.1)

        """
        Timeseries for paper
        """
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(40, 20), gridspec_kw={'height_ratios': [2, 2]})
        for i, row in grand_avg.iterrows():
            axes[0].scatter(row['date_ymd'], row['mean_ratio'], color=self.colors['dark_blue'], alpha=transp.iloc[i], s=15)
            axes[1].scatter(row['date_ymd'], row['mean_hate_ratio'], color=self.colors['purple'], alpha=transp.iloc[i], s=15)

            assert not (row['n_videos'] > 0 and row['n_users'] == 0)


        axes[0].plot(grand_avg['date_ymd'], grand_avg['mean_ratio'], color=self.colors['dark_blue'], linewidth=2.5)
        axes[0].fill_between(grand_avg['date_ymd'], grand_avg['mean_ratio'] - grand_avg['sem_ratio'], grand_avg['mean_ratio'] + grand_avg['sem_ratio'], color=self.colors['dark_blue'], alpha=0.15)
        axes[1].plot(grand_avg['date_ymd'], grand_avg['mean_hate_ratio'], color=self.colors['purple'], linewidth=2.5)
        axes[1].fill_between(grand_avg['date_ymd'], grand_avg['mean_hate_ratio'] - grand_avg['sem_hate_ratio'], grand_avg['mean_hate_ratio'] + grand_avg['sem_hate_ratio'], color=self.colors['purple'], alpha=0.15)

        legend = []
        for label, date in self.political_events.items():
            for i, plot in enumerate(axes):
                axes[i].axvline(date, linestyle='--', color=self.colors['green'], alpha=0.8)
                axes[i].axvspan(date - pd.Timedelta(days=14), date + pd.Timedelta(days=3), color=self.colors['green'], alpha=0.08)

            axes[1].text(date, -0.1, label, rotation=45, ha='right', va='top', fontsize=self.axesticks, color=self.colors['green'], transform=axes[1].get_xaxis_transform())

        axes[0].set_ylabel('Ratio News in Feed', fontsize=27)
        axes[0].set_title("Exposure to News, Politics & Societal Issues Content and Hate over Time\n", fontsize=self.fontsize_titles)

        axes[0].set_xlim(pd.to_datetime('2023-10-01'), pd.to_datetime('2026-01-01'))
        axes[0].set_ylim(0, 0.3)

        axes[1].set_ylabel('Ratio Hate\nNews in Feed', fontsize=self.fontsize_labels)
        axes[1].set_ylim(0, 0.3)

        for ax in axes:
            ax.tick_params(axis='both', labelsize=self.axesticks)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        ticks = pd.date_range(start='2023-10-01', end='2026-01-01', freq='9MS')
        axes[1].set_xticks(ticks)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.xticks(rotation=45)
        fig.subplots_adjust(bottom=0.25)

        div_by_channel_type()

        """
        Data availability for supplement
        """
        fig_supplement, axes2 = plt.subplots(2, 1, sharex=True, figsize=(40, 20))

        axes2[0].plot(grand_avg['date_ymd'], grand_avg['n_videos'], color=self.colors['brown'], linewidth=1.5, label='Number of Videos')
        axes2[0].set_ylabel('Number of\nVideos', fontsize=self.fontsize_labels)

        axes2[1].plot(grand_avg['date_ymd'], grand_avg['n_users'], color=self.colors['orange'], alpha=0.5, label='Number of Participants')
        axes2[1].set_ylabel('Number of\nParticipants', fontsize=self.fontsize_labels)
        
        for label, date in self.political_events.items():
            for i, plot in enumerate(axes2):
                axes2[i].axvline(date, linestyle='--', color=self.colors['green'], alpha=0.8)
                axes2[i].axvspan(date - pd.Timedelta(days=14), date + pd.Timedelta(days=3), color=self.colors['green'], alpha=0.08)

        for ax in axes2:
            ax.tick_params(axis='both', labelsize=self.axesticks)

        ticks2 = pd.date_range(start='2023-10-01', end='2026-01-01', freq='9MS')
        axes2[1].set_xticks(ticks2)
        axes2[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        axes2[0].set_title("Data Availability 2023-2026\n", fontsize=self.fontsize_titles)

        ticks = pd.date_range(start='2023-10-01', end='2026-01-01', freq='9MS')
        axes2[1].set_xticks(ticks)
        axes2[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        plt.xticks(rotation=45)
        fig.subplots_adjust(bottom=0.22)

        return fig, fig_supplement

    def get_graph_report(self, *figs: tuple[plt.Figure]) -> None:
        with PdfPages('graphs.pdf') as pdf:
            for fig in figs:
                pdf.savefig(fig) 

    def export_png(self,  *figs: tuple[plt.Figure], custom_label='') -> None:
        for i, fig in enumerate(figs):
            fig.savefig(f'plots/{custom_label}_{i}.png', dpi=300)

 
class DataExtract:
    def __init__(self, database: Database, base_dir=BASE_DIR, topicColumnName='topic', 
                 sentColumnName='sent', intoleranceColumnName='intolerance', impolitenessColumnName='impoliteness', 
                 subtopicColumnName='subtopic'):
        self.database: Database = database
        self.base_dir = base_dir

        self.topicColumnName: str = topicColumnName
        self.sentColumnName: str = sentColumnName
        self.intoleranceColumnName: str = intoleranceColumnName
        self.impolitenessColumnName: str = impolitenessColumnName
        self.subtopicColumnName: str = subtopicColumnName

    def request_time_series(self, channel_ordering: bool=False, get_qs: bool=False) -> list[pd.DataFrame]:
        feed_data: pd.DataFrame = self.database.retrieve_data(tableName='engagements e LEFT JOIN videos v ON e.video_id=v.id LEFT JOIN transcript_classification tc ON tc.video_id=e.video_id', 
                                                              colNames=['tc.video_id', f'tc.{self.topicColumnName}', f'tc.{self.sentColumnName}', 
                                                                        f'tc.{self.intoleranceColumnName}', f'tc.{self.impolitenessColumnName}', 'tc.video_duration', 'v.author_id', 'e.user_id', 'e.date_ymd', 'e.time_hms'], 
                                                              restrictions='.user_id NOT IN (119, 132)') # participant 119 was excluded due to being to old (age 30), 132 was excluded due to having donated only one video from 2020

        feed_data = self.format_time(feed_data)
        feed_data = feed_data[feed_data['topic'].notna()]

        official, unofficial, ordering = self.load_channels()
        channels: pd.DataFrame = self.database.retrieve_data(tableName='authors', colNames=['id AS author_id', 'unique_id'])
        channels['official'] = np.select([channels['unique_id'].isin(official), channels['unique_id'].isin(unofficial)], [1, 2], default=3)

        feed_data = feed_data.merge(channels, on='author_id', how='left')
        feed_data['high_hate'] = ((feed_data[self.intoleranceColumnName] >= 5) | (feed_data[self.impolitenessColumnName] >= 5)).astype(int)


        feed_data['is_news'] = (feed_data[self.topicColumnName] == TOPIC).astype(int)
        feed_data['date_ymd'] = pd.to_datetime(feed_data['date_ymd'])
        feed_data = feed_data.sort_values(['user_id', 'date_ymd', 'time_hms'])

        if get_qs:
            qs = self.database.retrieve_data(tableName='questionnaires', colNames=['id AS user_id', 'age'])
            feed_data = feed_data.merge(qs, on='user_id', how='left')
            feed_data = self.rollback_age(feed_data)

        feed_data_dfs = [feed_data.copy()]
        if channel_ordering:
            for n in range(len(ordering)):
                def_df = feed_data.copy() 

                def_df.loc[def_df['unique_id'].isin(ordering[str(n)]), 'official'] = 2 
                feed_data_dfs.append(def_df)

                # assert def_df.loc[def_df['unique_id'] == 'foxnews', 'official'].iloc[0] == (1 if n < 8 else 2)

        return feed_data_dfs

 
    @staticmethod
    def format_time(df: pd.DataFrame, dateColumnName: str='date_ymd', timeColumnName: str='time_hms') -> pd.DataFrame: 
        df['timestamp'] = pd.to_datetime(df[dateColumnName] + ' ' + df[timeColumnName])
        df = df.sort_values(['user_id', 'timestamp'])
        df['next_ts'] = df.groupby('user_id')['timestamp'].shift(-1)
        df['watch_time_s'] = (df['next_ts'] - df['timestamp']).dt.total_seconds().fillna(0)
        
        # RD: a new session is defined after 7.5 min without scrolling
        df['new_session'] = (df['watch_time_s'] >= 450).astype(int)
        df.loc[df.groupby('user_id').tail(1).index, 'new_session'] = 2
        
        return df

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

class Robustness:
    def __init__(self, database: Database, topicColumnName='topic', intoleranceColumnName='topic', impolitenessColumnName='impoliteness'):
        self.topicColumnName: str = topicColumnName
        self.intoleranceColumnName: str = intoleranceColumnName
        self.impolitenessColumnName: str = impolitenessColumnName

        self.database: Database = database

    def robustness_hate_source(self, dfs):
        definitions = ['user_exposures']# , 'reach', 'avg_users']

        for label in definitions:
            self.get_top500(dfs, label)
 
    def get_top500(self, dfs, label):
        stats = Statistics()
        graphs = Graphs()

        dfs = [self.switch_alternative_label(df, label) for df in dfs]

        robust_hate_source = graphs.create_hate_source_graph(dfs)
        # follower_graph = self.get_followers(dfs[0])
        graphs.export_png(robust_hate_source, custom_label=f'robust_{label}')

        stats.get_hate_source(dfs)
        stats.dump_to_json(label=f'robustness/{label}')

    def switch_alternative_label(self, df, label):
        official = df.loc[df['official'] == 1, 'unique_id'].to_list()
        official_sql = "(" + ",".join(f"'{x}'" for x in set(official)) + ")"

        unofficial = df.loc[df['official'] == 2, 'unique_id'].to_list()
        unofficial_sql = "(" + ",".join(f"'{x}'" for x in set(unofficial)) + ")"
        print(unofficial_sql)

        if label == 'user_exposures':
            definition = self.define_exposure(official_sql)
            print(official_sql)
        elif label == 'reach':
            definition = self.define_reach(official_sql)
        elif label == 'followers':
            definition = self.define_followers(official_sql)
        else:
            definition = self.define_avg_audience(official_sql)

        with sqlite3.connect(BASE_DIR/'database/150_db.sqlite3') as db:
            top_channels = pd.read_sql(definition, db)

        top_channel_ids = set(top_channels['unique_id'])

        local_df = df.copy()
        local_df.loc[local_df['unique_id'].isin(top_channel_ids), 'official'] = 2
        local_df.loc[(~local_df['unique_id'].isin(top_channel_ids)) & (local_df['official'] == 2), 'official'] = 3

        get_followers = self.database.retrieve_data(tableName='authors', colNames=['id AS author_id', 'follower_count'])
        local_df = local_df.merge(get_followers, on='author_id', how='left')
        local_df.loc[(local_df['official'] == 3) & (local_df['follower_count'] > local_df.loc[local_df['official'] == 2, 'follower_count'].quantile(0.90)), 'official'] = 2

        # local_df.loc[(local_df['official'] == 3) & (local_df['follower_count'] > 338000), 'official'] = 2 # Move all channels with > 338000 followers to alternative
        return local_df

    # Popularity is defined as user exposures: each unique combination of a user and watched news vid from a channel is 1 exposure
    @staticmethod
    def define_exposure(official_sql):
        return f"""
            SELECT unique_id, follower_count, COUNT(*) AS user_video_exposures FROM (
                SELECT DISTINCT e.user_id, e.video_id, a.unique_id, a.follower_count
                FROM engagements e JOIN videos v ON e.video_id = v.id
                JOIN transcript_classification tc ON e.video_id = tc.video_id
                JOIN authors a ON v.author_id = a.id WHERE tc.topic = 'News, Politics & Societal Issues')
            WHERE unique_id NOT IN {official_sql}
            GROUP BY unique_id
            ORDER BY user_video_exposures DESC
            LIMIT 500;
        """

    # popularity is defined as reach -> how many unique users did this channels reach with their news videos
    @staticmethod
    def define_reach(official_sql):
        return f"""
            SELECT unique_id, follower_count, COUNT(DISTINCT user_id) AS unique_users FROM (
                SELECT DISTINCT e.user_id, e.video_id, a.unique_id, a.follower_count
                FROM engagements e 
                JOIN videos v ON e.video_id = v.id
                JOIN transcript_classification tc ON e.video_id = tc.video_id
                JOIN authors a ON v.author_id = a.id WHERE tc.topic = 'News, Politics & Societal Issues')
            WHERE unique_id NOT IN {official_sql}
            GROUP BY unique_id
            ORDER BY unique_users DESC
            LIMIT 500;
            """

    # popularity as defined by average users that watched each news video from the channel
    @staticmethod
    def define_avg_audience(official_sql):
       return f"""
            SELECT unique_id, follower_count, CAST(COUNT(*) AS FLOAT) / COUNT(DISTINCT video_id) AS avg_users_per_video
            FROM (SELECT DISTINCT e.user_id, e.video_id, a.unique_id, a.follower_count
                FROM engagements e 
                JOIN videos v ON e.video_id = v.id
                JOIN transcript_classification tc ON e.video_id = tc.video_id
                JOIN authors a ON v.author_id = a.id WHERE tc.topic = 'News, Politics & Societal Issues')
            WHERE unique_id NOT IN {official_sql}
            GROUP BY unique_id
            ORDER BY avg_users_per_video DESC
            LIMIT 500;
        """         

    def get_followers(self, df):
        high_hate: pd.DataFrame = df[df['high_hate'] == 1]

        alternative_sources = high_hate[high_hate['official'] == 2]
        niche_sources = high_hate[high_hate['official'] == 3]

        get_followers = self.database.retrieve_data(tableName='authors', colNames=['id AS author_id', 'follower_count'])
        alternative_sources = alternative_sources.merge(get_followers, on='author_id', how='left').sort_values(by=['follower_count'], ascending=False)
        niche_sources = niche_sources.merge(get_followers, on='author_id', how='left').sort_values(by=['follower_count'], ascending=False)

        fig, ax = plt.subplots(1, 1, figsize=(40, 20))
        
        x_alt = np.arange(len(alternative_sources))
        x_niche = np.arange(len(niche_sources)) + len(alternative_sources) + 5
                
        ax.bar(x_alt, alternative_sources['follower_count'], label='alternative')
        ax.bar(x_niche, niche_sources['follower_count'], label='niche')

        ax.set_ylim(min(alternative_sources['follower_count'].min(), niche_sources['follower_count'].min()), 
                    max(alternative_sources['follower_count'].max(), niche_sources['follower_count'].max()) * 1.1)

        ax.text(x_alt.mean(), 
                max(alternative_sources['follower_count'].max(), niche_sources['follower_count'].max()) * 1.1, 
                f"Median: {alternative_sources['follower_count'].median():,.0f}",ha="center", fontsize=14, fontweight="bold")

        ax.text(x_niche.mean(), 
                max(alternative_sources['follower_count'].max(), niche_sources['follower_count'].max()) * 1.1, 
                f"Median: {niche_sources['follower_count'].median():,.0f}",ha="center", fontsize=14, fontweight="bold")

        ax.set_xticks([])

        return fig

        
     
if __name__=='__main__':
    main()
