import pandas as pd
from database import *
import json
from pathlib import Path
import seaborn as sns
import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from scipy.stats import f_oneway, pearsonr
from statsmodels.stats.multicomp import MultiComparison
from itertools import combinations
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent


class Hate_Speech_Target:
    def __init__(self, subtopic_file=BASE_DIR/"src/keywords3.json"):
        with open(subtopic_file, "r") as f:
            self.keywords = json.load(f)["keywords"]
        self.columns = ["user_id", "channel_type", "transcript_count"] + list(self.keywords.keys()) + ["channel_ratio"]
        self.axesticks = 26
        self.fontsize_labels = 34
        self.fontsize_titles = 40

    def get_hate_speech(self):
        db = Database(db_path=BASE_DIR/'database/150_db.sqlite3')

        self.data = db.retrieve_data(
            tableName='engagements e LEFT JOIN transcript_classification tc ON e.video_id = tc.video_id LEFT JOIN videos v ON tc.video_id = v.id LEFT JOIN authors a ON a.id=v.author_id',
            colNames=['e.user_id', 'tc.translation', 'v.desc', 'unique_id'], restrictions='(tc.impoliteness >= 5 OR tc.intolerance >= 5) AND e.user_id NOT IN (119, 132)')

        official, unofficial, ordering = self.load_channels()

        self.data['channel_type'] = np.select([self.data['unique_id'].isin(official), self.data['unique_id'].isin(unofficial)], [1, 2], default=3)
        self.data.loc[self.data['unique_id'].isin(ordering[str(66)]), 'channel_type'] = 2

        self.data['translation'] = self.data['translation'].fillna("").astype(str)
        self.data['desc'] = self.data['desc'].fillna("").astype(str)
        self.data['text'] = self.data['translation'] + " " + self.data['desc']

    def _match_keywords(self, text, keywords):
        text_lower = text.lower()
        for keyword in keywords:
            if keyword.lower() in text_lower:
                return 1
        return 0

    def add_keyword_columns(self):
        for subtopic, keywords in self.keywords.items():
            self.data[subtopic] = self.data['text'].apply(lambda text: self._match_keywords(text, keywords))

    def compute_user_channel_ratios(self):
        grouped = self.data.groupby(['user_id', 'channel_type']).agg(transcript_count=('user_id', 'size'),
            **{subtopic: (subtopic, 'sum') for subtopic in self.keywords.keys()}).reset_index()

        all_users = grouped['user_id'].unique()

        channel_types = self.data['channel_type'].unique()
        index = pd.MultiIndex.from_product([all_users, channel_types], names=['user_id', 'channel_type'])
        grouped = grouped.set_index(['user_id', 'channel_type']).reindex(index).reset_index()

        grouped['transcript_count'] = grouped['transcript_count'].fillna(0)
        subtopic_cols = list(self.keywords.keys())
        grouped[subtopic_cols] = grouped[subtopic_cols].fillna(0)

        subtopic_sums = grouped[subtopic_cols].sum(axis=1)
        for subtopic in subtopic_cols:
            grouped[subtopic] = grouped[subtopic] / subtopic_sums

        user_totals = grouped.groupby('user_id')['transcript_count'].transform('sum')
        grouped['channel_ratio'] = grouped['transcript_count'] / user_totals
        return grouped[self.columns]

    def run_anovas(self, ratios: pd.DataFrame):
        channel_data = {}

        for ch in [1, 2, 3]:
            vals = ratios.loc[ratios['channel_type'] == ch, 'channel_ratio'].dropna().tolist()

            if len(vals) >= 2:
                channel_data[ch] = vals

        channel_groups = [v for v in channel_data.values() if len(v) > 0]

        if len(channel_groups) >= 3:
            f_stat, channel_p = f_oneway(*channel_groups)
            k, n = len(channel_groups), sum(len(g) for g in channel_groups)
            df_between = k - 1
            df_within = n - k
            print(f"Channel ANOVA: F={f_stat:.4f}, p={channel_p:.6f}, df_between={df_between}, df_within={df_within}")

        else:
            channel_p = None

        self._posthoc_pairwise(ratios, 'channel_type', 'channel_ratio')

        subtopic_cols = list(self.keywords.keys())
        user_subtopic = ratios.groupby('user_id')[subtopic_cols].sum().reset_index()
        subtopic_data = {col: user_subtopic[col].dropna().tolist() for col in subtopic_cols}
        subtopic_groups = [v for v in subtopic_data.values() if len(v) > 0]

        if len(subtopic_groups) >= 3:
            f_stat, subtopic_p = f_oneway(*subtopic_groups)
            k_sub, n_sub = len(subtopic_groups), sum(len(g) for g in subtopic_groups)
            df_between_sub = k_sub - 1
            df_within_sub = n_sub - k_sub
            print(f"Subtopic ANOVA: F={f_stat:.4f}, p={subtopic_p:.6f}, df_between={df_between_sub}, df_within={df_within_sub}")

        else:
            subtopic_p = None

        subtopic_tall = user_subtopic.melt(id_vars=[], value_vars=subtopic_cols, var_name='subtopic', value_name='value')

        self._posthoc_pairwise(subtopic_tall, 'subtopic', 'value')

        return {'channel_f_stat': f_stat if len(channel_groups) >= 3 else None, 'channel_p': channel_p,
                'subtopic_f_stat': f_stat if len(subtopic_groups) >= 3 else None, 'subtopic_p': subtopic_p}

    @staticmethod
    def _posthoc_pairwise(df, group_col, value_col):
        groups = df.groupby(group_col)[value_col].apply(list)
        groups = {k: v for k, v in groups.items() if len(v) >= 2} 

        all_values, all_labels = [], []

        for label, vals in groups.items():
            all_values.extend(vals)
            all_labels.extend([label] * len(vals))

        mc = MultiComparison(np.array(all_values), np.array(all_labels))
        print(mc.tukeyhsd().summary())

    @staticmethod
    def load_channels(channelDefinitionFileName: str = BASE_DIR / 'src' / 'channel_def.json') -> tuple:
        with open(channelDefinitionFileName, 'r') as f:
            channels = json.load(f)

        return channels['official'], channels['unofficial'], channels['ordering']

    def plot_channel_ratios(self, ratios):
        channel_colors = {1: '#001157', 2: '#b03aa3', 3: '#ec8e3a'}
        channel_labels = {1: 'Legacy', 2: 'Alternative', 3: 'Niche'}
        channel_order = [1, 2, 3]

        categories = list(self.keywords.keys())
        subtopic_cols = list(self.keywords.keys())

        user_avg = ratios.groupby(['user_id', 'channel_type'])[subtopic_cols + ['channel_ratio']].mean()
        channel_df = user_avg.groupby('channel_type').mean().reset_index()

        subtopic_df = (user_avg[subtopic_cols].groupby(user_avg.index.get_level_values('channel_type')).mean().reset_index().melt(id_vars='channel_type', var_name='label', value_name='ratio'))

        rows = []

        for channel in channel_order:
            ch_df = channel_df[channel_df['channel_type'] == channel]
            channel_ratio = ch_df['channel_ratio'].values[0]
            total_subtopics = ch_df[subtopic_cols].sum(axis=1).values[0]

            for cat in categories:
                original = ch_df[cat].values[0]
                ratio = (original / total_subtopics) * channel_ratio if total_subtopics > 0 and channel_ratio > 0 else 0
                rows.append({"label": cat, "channel_type": channel, "ratio": ratio, "original_ratio": original})

        df = pd.DataFrame(rows)

        y_positions, current_y, channel_positions = [], 0, {}
        for channel in channel_order:
            start = current_y
            for _ in range(len(categories)):
                y_positions.append(current_y)

                current_y += 1
            channel_positions[channel] = (start, current_y - 1)
            current_y += 1

        fig, ax = plt.subplots(figsize=(40, 26))

        channel_bar_values = {}
        for channel, (start, end) in channel_positions.items():

            midpoint = (start + end) / 2
            ch_ratio = channel_df.loc[channel_df['channel_type'] == channel, 'channel_ratio'].values[0]
            channel_bar_values[channel] = ch_ratio
            ax.barh(midpoint, ch_ratio, height=(end - start) + 1.15, color=channel_colors[channel], alpha=0.14, zorder=0)

        for y, (_, row) in zip(y_positions, df.iterrows()):

            ax.barh(y, row["ratio"], color=channel_colors[row["channel_type"]], height=0.60, zorder=2)

        max_bar = max(channel_bar_values.values())
        ax.set_xlim(0, max_bar)
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()

        for channel, (start, end) in channel_positions.items():
            midpoint = (start + end) / 2
            ch_ratio = channel_df.loc[channel_df['channel_type'] == channel, 'channel_ratio'].values[0]
            label = ax.text(ch_ratio + 0.01, midpoint, channel_labels[channel], transform=ax.transData, ha="left", va="center", fontsize=self.axesticks, fontweight="bold")

            ext = label.get_window_extent(renderer=renderer)
            pt_to_data_x = 1.0 / (ax.transData.transform([[1, 0]])[0][0] - ax.transData.transform([[0, 0]])[0][0])
            ax.text(ch_ratio + 0.01 + ext.width * pt_to_data_x + 0.025, midpoint, f"{ch_ratio*100:.1f}%", transform=ax.transData, ha="left", va="center", fontsize=self.axesticks, fontweight="normal")

        ax.set_xticks(MaxNLocator(nbins=10).tick_values(0, max_bar))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df["label"], fontsize=self.axesticks)
        ax.invert_yaxis()

        ax.set_xlabel("\nAverage Share of Total Hate (%)", fontsize=self.fontsize_labels)
        ax.set_title("\nAverage Share of Total Hate by Channel Type,\n Broken Up by Subtopic", fontsize=self.fontsize_titles, pad=18)
        ax.tick_params(axis="x", labelsize=self.axesticks)

        sns.despine(left=True)

        ax.legend(handles=[Patch(color=channel_colors[ch], label=channel_labels[ch]) for ch in channel_order], loc='lower right', fontsize=self.axesticks, frameon=False)
        plt.subplots_adjust(left=0.28, right=0.95, top=0.88, bottom=0.15)

        fig.savefig("plots/hate_channel_type_ratios.png", dpi=300, bbox_inches="tight", pad_inches=0.25)
        plt.show()
        return fig


class Correlation_Matrix:
    def __init__(self):
        self.db = Database(db_path=BASE_DIR/'database/150_db.sqlite3')
        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}

        self.fontsize_labels: int = 30
        self.fontsize_titles: int = 36
        self.axesticks: int = 26

    def get_data(self) -> pd.DataFrame:
        q_items = ['id', 'age', 'political_preference', 'dimi1_compulsion1', 'future_anxiety_1', 'future_anxiety_2', 'future_anxiety_3',
                      'future_anxiety_4', 'future_anxiety_5', 'sr_hate_speech', 'affective_pol', 'future_anxiety_total', 'dimi_compulsion_total', 
                      'dimi_choice_total', 'boredom_total', 'skills_self_control_total', 'skills_misinformation_total',
                        'dimi2_compulsion2', 'dimi3_compulsion3', 'dimi4_choice1','dimi5_choice2', 'dimi6_choice3', 'skills1_self_control1',
                        'skills2_self_control2', 'skills3_self_control3', 'skills4_misinformation1', 'skills5_misinformation2',
                        'skills6_misinformation3', 'boredom1', 'boredom2', 'boredom3', 'news_1', 'news_3', 'news_4_1',
                        'news_4_2', 'ipsos_statements_1', 'ipsos_statements_2', 'ipsos_statements_3', 'ipsos_statements_4',
                        'ipsos_statements_5', 'ipsos_statements_6', 'ipsos_statements_7', 'ipsos_statements_8', 'ipsos_statements_1_opinion_extreme', 
                        'ipsos_statements_2_opinion_extreme', 'ipsos_statements_3_opinion_extreme', 'ipsos_statements_4_opinion_extreme',
                        'ipsos_statements_5_opinion_extreme', 'ipsos_statements_6_opinion_extreme', 'ipsos_statements_7_opinion_extreme', 'ipsos_statements_8_opinion_extreme']

        query = f"""
        SELECT user_id, hate_ratio, news_ratio {', '.join(q for q in q_items)}
        FROM(
            SELECT AVG(
                CASE WHEN tc.intolerance >= 5 OR tc.impoliteness >= 5 THEN 1
                WHEN tc.topic != 'News, Politics & Societal Issues' THEN NULL
                ELSE 0
            END) AS hate_ratio,
            AVG(
                CASE WHEN tc.topic='News, Politics & Societal Issues' THEN 1
                ELSE 0
            END) AS news_ratio, e.user_id AS user_id
            FROM engagements e
            LEFT JOIN transcript_classification tc ON e.video_id=tc.video_id
            WHERE tc.topic IS NOT NULL AND e.user_id NOT IN (119, 132)
            GROUP BY e.user_id
        )
        LEFT JOIN questionnaires q ON q.id=user_id;
        """

        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        return df

    def specific_cor(self, df: pd.DataFrame):
        df = df[['hate_ratio', 'sr_hate_speech']].dropna()
        r, p = pearsonr(df['hate_ratio'], df['sr_hate_speech'])

        print('r', r)
        print('p', p)


if __name__ == '__main__':
    target = Hate_Speech_Target()
    target.get_hate_speech()
    target.add_keyword_columns()

    ratios = target.compute_user_channel_ratios()
    results = target.run_anovas(ratios)

    target.plot_channel_ratios(ratios)

    cor = Correlation_Matrix()
    df = cor.get_data()
    cor.specific_cor(df)


