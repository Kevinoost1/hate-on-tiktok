import pandas as pd
from database import *
import json
from pathlib import Path
import seaborn as sns
import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from scipy.stats import f_oneway, pearsonr, shapiro, wilcoxon
from statsmodels.stats.multicomp import MultiComparison
from itertools import combinations
import numpy as np
import zstandard as zstd

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

        channel_sems = user_avg['channel_ratio'].groupby(
            user_avg.index.get_level_values('channel_type')).sem()

        subtopic_contributions = user_avg[subtopic_cols].copy()
        for col in subtopic_cols:
            subtopic_contributions[col] = subtopic_contributions[col] * user_avg['channel_ratio']

        subtopic_sems = subtopic_contributions.groupby(
            subtopic_contributions.index.get_level_values('channel_type')).sem()


        subtopic_df = (user_avg[subtopic_cols].groupby(user_avg.index.get_level_values('channel_type')).mean().reset_index().melt(id_vars='channel_type', var_name='label', value_name='ratio'))

        rows = []

        for channel in channel_order:
            ch_df = channel_df[channel_df['channel_type'] == channel]
            channel_ratio = ch_df['channel_ratio'].values[0]
            total_subtopics = ch_df[subtopic_cols].sum(axis=1).values[0]

            for cat in categories:
                original = ch_df[cat].values[0]
                ratio = (original / total_subtopics) if total_subtopics > 0 and channel_ratio > 0 else 0
                rows.append({"label": cat, "channel_type": channel, "ratio": ratio, "original_ratio": original})

        df = pd.DataFrame(rows)

        y_positions, current_y, channel_positions = [], 0, {}
        for channel in channel_order:
            start = current_y
            for _ in range(len(categories)):
                y_positions.append(current_y)

                current_y += 1
            channel_positions[channel] = (start, current_y - 1)
            current_y += 1.8

        fig, ax = plt.subplots(figsize=(40, 26))

        channel_bar_values = {}
        for channel, (start, end) in channel_positions.items():

            midpoint = (start + end) / 2
            ch_ratio = channel_df.loc[channel_df['channel_type'] == channel, 'channel_ratio'].values[0]
            channel_bar_values[channel] = ch_ratio
            ax.barh(midpoint, ch_ratio, height=(end - start) + 1.50, facecolor=(*plt.matplotlib.colors.to_rgb(channel_colors[channel]), 0.14), zorder=0, 
                        edgecolor=(*plt.matplotlib.colors.to_rgb(channel_colors[channel]), 0.45), linewidth=2)
            ax.errorbar(ch_ratio, midpoint, xerr=channel_sems[channel], fmt='none',
                        ecolor=channel_colors[channel], capsize=8, capthick=2, zorder=1)

        for y, (_, row) in zip(y_positions, df.iterrows()):

            ax.barh(y, row["ratio"], color=channel_colors[row["channel_type"]], height=0.60, zorder=2, alpha=0.9)
            sem_val = subtopic_sems.loc[row["channel_type"], row["label"]]
            ax.errorbar(row["ratio"], y, xerr=sem_val, fmt='none',
                        ecolor=channel_colors[row["channel_type"]], capsize=8, capthick=2, zorder=3, alpha=0.9)


        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()

        for channel, (start, end) in channel_positions.items():
            midpoint = (start + end) / 2
            ch_ratio = channel_df.loc[channel_df['channel_type'] == channel, 'channel_ratio'].values[0]
            label = ax.text(0.75, midpoint - 1, channel_labels[channel], transform=ax.transData, ha="left", va="center", fontsize=self.axesticks, fontweight="bold")

            ext = label.get_window_extent(renderer=renderer)
            pt_to_data_x = 1.0 / (ax.transData.transform([[1, 0]])[0][0] - ax.transData.transform([[0, 0]])[0][0])
            ax.text(0.80 + ext.width * pt_to_data_x + 0.025, midpoint - 1, f"{ch_ratio*100:.1f}%", transform=ax.transData, ha="left", va="center", fontsize=self.axesticks, fontweight="normal")

        ax.set_xticks(MaxNLocator(nbins=10).tick_values(0, 1))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val*100:.0f}%"))
        ax.set_yticks(y_positions)
        ax.set_yticklabels(df["label"], fontsize=self.axesticks)
        ax.invert_yaxis()

        ax.set_xlabel("\nAverage Percentage of Total Hate (%)", fontsize=self.fontsize_labels)
        ax.set_title("\nAverage Percentage of Total Hate by Channel Type,\n Broken Up by Subtopic", fontsize=self.fontsize_titles, pad=18)
        ax.tick_params(axis="x", labelsize=self.axesticks)

        sns.despine(left=True)
        ax.set_xlim(0, 1)

        ax.legend(handles=[Patch(color=channel_colors[ch], label=channel_labels[ch]) for ch in channel_order], loc='lower right', fontsize=self.axesticks, frameon=False)
        plt.subplots_adjust(left=0.28, right=0.95, top=0.88, bottom=0.15)

        fig.savefig("plots/hate_channel_type_ratios.png", dpi=300, bbox_inches="tight", pad_inches=0.25)
        plt.show()
        return fig



    def plot_channel_ratios_fourPanel(self, ratios):
        channel_colors = {1: '#001157', 2: '#b03aa3', 3: '#ec8e3a'}
        channel_labels = {1: 'Legacy', 2: 'Alternative', 3: 'Niche'}
        channel_order = [1, 2, 3]

        categories = list(self.keywords.keys())
        subtopic_cols = list(self.keywords.keys())

        user_avg = ratios.groupby(['user_id', 'channel_type'])[subtopic_cols + ['channel_ratio']].mean()

        channel_df = (user_avg.groupby('channel_type').mean().reset_index())
        channel_sems = user_avg['channel_ratio'].groupby(user_avg.index.get_level_values('channel_type')).sem()

        subtopic_contributions = user_avg[subtopic_cols].copy()

        for col in subtopic_cols:
            subtopic_contributions[col] = (
                subtopic_contributions[col] * user_avg['channel_ratio'])

        subtopic_sems = subtopic_contributions.groupby(subtopic_contributions.index.get_level_values('channel_type')).sem()

        rows = []
        for channel in channel_order:
            ch_df = channel_df[
                channel_df['channel_type'] == channel]

            for cat in categories:
                original = ch_df[cat].values[0]

                rows.append({'label': cat, 'channel_type': channel, 'ratio': original, 'original_ratio': original})

        df = pd.DataFrame(rows)
        fig, axes = plt.subplots(2, 2, figsize=(40, 28), sharex=True)

        ax_channel = axes[0, 0]
        subtopic_axes = [
            axes[0, 1],  # Legacy
            axes[1, 0],  # Alternative
            axes[1, 1]   # Niche
        ]

        ax = ax_channel
        for y, channel in enumerate(channel_order):
            value = channel_df.loc[channel_df['channel_type'] == channel, 'channel_ratio'].values[0]

            color_rgb = plt.matplotlib.colors.to_rgb(channel_colors[channel])

            ax.barh(y, value, height=0.60, facecolor=(*color_rgb, 1.0), edgecolor=(*color_rgb, 1.0), linewidth=1.5, zorder=2)
            ax.errorbar(value, y, xerr=channel_sems[channel], fmt='none', ecolor=channel_colors[channel], capsize=8, capthick=2, zorder=3)

            ax.annotate(f'{value * 100:.2f}%', xy=(value + 0.05, y), xytext=(8, 0), textcoords='offset points', ha='left', va='center', 
                        fontsize=self.axesticks, clip_on=False)

        ax.set_yticks(range(len(channel_order)))
        ax.set_yticklabels([channel_labels[ch] for ch in channel_order], fontsize=self.axesticks, fontweight='bold')

        ax.invert_yaxis()
        ax.set_title('Average Percentage of Total Hate by Channel Type', fontsize=self.fontsize_titles, pad=24)

        for ax, channel in zip(subtopic_axes, channel_order): 
            channel_data = (df[df['channel_type'] == channel].set_index('label').reindex(categories).reset_index())

            y_positions = list(range(len(categories)))
            color_rgb = plt.matplotlib.colors.to_rgb(channel_colors[channel])

            for y, (_, row) in zip(y_positions, channel_data.iterrows()):

                ax.barh(y, row['ratio'], height=0.60, facecolor=(*color_rgb, 0.70), edgecolor=(*color_rgb, 1.0), linewidth=2.5, zorder=2)

                sem_val = subtopic_sems.loc[channel, row['label']]

                ax.errorbar(row['ratio'], y, xerr=sem_val, fmt='none', ecolor=channel_colors[channel], capsize=8, capthick=2, zorder=3)

            ax.set_yticks(y_positions)
            ax.set_yticklabels(channel_data['label'], fontsize=self.axesticks)
            ax.invert_yaxis()
            ax.set_title(f' Subtopic Distribution Within {channel_labels[channel]}', fontsize=self.fontsize_titles, pad=24)

        for ax in axes.flat:
            ax.set_xlim(0, 1)
            ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f'{val * 100:.0f}%'))

            ax.tick_params(axis='x', labelsize=self.axesticks, labelbottom=True, pad=10)
            ax.tick_params(axis='y', labelsize=self.axesticks, pad=10)
            ax.set_xlabel('Average Percentage of Total Hate (%)', fontsize=self.fontsize_labels, labelpad=14)

            sns.despine(ax=ax, left=True)
            ax.margins(x=0.03, y=0.12)

        plt.subplots_adjust(left=0.18, right=0.96, top=0.94, bottom=0.09, wspace=0.80, hspace=0.65)

        fig.canvas.draw()
        fig.savefig('plots/hate_channel_type_ratios_fourPanel.png', dpi=300)
        plt.show()

        return fig


    def export_tiff(self, *figs: tuple[plt.Figure], custom_label='hateSource') -> None:
        for i, fig in enumerate(figs):
            fig.savefig(f'plots/{custom_label}_{i}.tiff', format='tiff', dpi=600)





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


class Cross_Ref_Followers():
    def __init__(self):
        self.db = Database(db_path=BASE_DIR/'database/150_db.sqlite3')


    def merge_following_vs_non_following(self, user_engagements):
        user_follower_list = self._compile_follower_list()

        is_following = set(zip(user_follower_list['subject'], user_follower_list['username']))
        user_engagements['is_following'] = user_engagements.apply(lambda x: int((x['subject'], x['username']) in is_following), axis=1)

        official, unofficial, ordering = self.load_channels()       
        user_engagements['channel_type'] = np.select([user_engagements['username'].isin(official), user_engagements['username'].isin(unofficial)], [1, 2], default=3)
        user_engagements.loc[user_engagements['username'].isin(ordering[str(66)]), 'channel_type'] = 2

        return user_engagements

    def _compile_follower_list(self):
        return pd.read_csv('following.csv')

    def get_engagements(self, analysis='tc.topic = "News, Politics & Societal Issues"'):
        query = f"""
        SELECT a.unique_id AS username, u.name AS subject, e.user_id, COUNT(e.video_id) AS video_count
        FROM engagements e
        LEFT JOIN videos v ON v.id = e.video_id
        LEFT JOIN authors a ON a.id = v.author_id
        LEFT JOIN transcript_classification tc ON tc.video_id = e.video_id
        LEFT JOIN users u ON u.id = e.user_id
        WHERE ({analysis}) AND e.user_id NOT IN (119, 132)
        GROUP BY e.user_id, a.unique_id
        ORDER BY e.user_id, video_count DESC;
        """

        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        return df

    def compute_ratios(self, user_engagements): 
        df = user_engagements.groupby('user_id').agg(
            video_count_is_following=('video_count', lambda x: user_engagements.loc[x.index].loc[user_engagements.loc[x.index, 'is_following'] == 1, 'video_count'].sum()),
            video_count_not_following=('video_count', lambda x: user_engagements.loc[x.index].loc[user_engagements.loc[x.index, 'is_following'] == 0, 'video_count'].sum()),
            total_video_count=('video_count', 'sum')
        ).reset_index()

        df['is_following_ratio'] = df['video_count_is_following'] / df['total_video_count']
        df['not_following_ratio'] = df['video_count_not_following'] / df['total_video_count']
 
        return df

    def compute_ratios_channel_types(self, user_engagements):
        channel_counts = user_engagements.groupby(['user_id', 'channel_type']).size().reset_index(name='count_following')
        total_counts = user_engagements.groupby('user_id').size().reset_index(name='total')

        df = channel_counts.merge(total_counts, on='user_id')
        df['ratio_following'] = df['count_following'] / df['total']

        print(df.groupby('channel_type').mean())

        return df


    def execute_wilcoxon_test(self, per_user):
        """Wilcoxon signed-rank test comparing is_following_ratio vs not_following_ratio."""
        valid = per_user.dropna(subset=['is_following_ratio', 'not_following_ratio'])

        stat, p_val = wilcoxon(valid['is_following_ratio'], valid['not_following_ratio'])

        diffs = valid['is_following_ratio'] - valid['not_following_ratio']
        shapiro_stat, shapiro_p = shapiro(diffs)

        print("=== Wilcoxon signed-rank test ===")
        print(f"Median is_following_ratio:     {valid['is_following_ratio'].median():.5f}")
        print(f"Max is_following_ratio:     {valid['is_following_ratio'].max():.5f}")
        print(f"Min is_following_ratio:     {valid['is_following_ratio'].min():.5f}")

        print(f"Median not_following_ratio:    {valid['not_following_ratio'].median():.5f}")
        print(f"Max not_following_ratio:     {valid['not_following_ratio'].max():.5f}")
        print(f"Min not_following_ratio:     {valid['not_following_ratio'].min():.5f}")

        print(f"V-statistic:                 {stat:.4f}")
        print(f"p-value:                     {p_val:.4f}")
        print(f"\n=== Normality of differences (Shapiro-Wilk) ===")
        print(f"Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")

        self.plot_ratio_histogram(valid)

        return {'stat': stat, 'p_val': p_val,
                'shapiro_stat': shapiro_stat, 'shapiro_p': shapiro_p,
                'n_valid': len(valid)}

    def plot_ratio_histogram(self, per_user):
        """Histogram of both ratios to check approximate normality."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        colors = {'is_following_ratio': '#001157', 'not_following_ratio': '#ec8e3a'}
        titles = {'is_following_ratio': 'is_following_ratio',
                  'not_following_ratio': 'not_following_ratio'}

        for i, (col, ax) in enumerate(zip(colors, axes)):
            valid = per_user[col].dropna()
            ax.hist(valid, bins=40, color=colors[col], edgecolor='white', alpha=0.85, density=True)

            kde = sns.kdeplot(x=valid, color='black', linewidth=1.5, ax=ax)
            ax.axvline(valid.mean(), color='red', linestyle='--', linewidth=1.5,
                       label=f'mean = {valid.mean():.3f}')
            ax.set_title(titles[col], fontsize=14)
            ax.set_xlabel('Ratio')
            ax.set_ylabel('Density')
            ax.legend()

            stat, p = shapiro(valid)
            ax.text(0.05, 0.95, f'Shapiro-Wilk\nstat={stat:.3f}\np={p:.3f}',
                    transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('plots/ratio_histograms.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def load_channels(channelDefinitionFileName: str = BASE_DIR / 'src' / 'channel_def.json') -> tuple:
        with open(channelDefinitionFileName, 'r') as f:
            channels = json.load(f)

        return channels['official'], channels['unofficial'], channels['ordering']



if __name__ == '__main__':
    target = Hate_Speech_Target()
    target.get_hate_speech()
    target.add_keyword_columns()

    ratios = target.compute_user_channel_ratios()
    results = target.run_anovas(ratios)

    hate_source_graph = target.plot_channel_ratios_fourPanel(ratios)
    target.export_tiff(hate_source_graph)

    # cor = Correlation_Matrix()
    # df = cor.get_data()
    # cor.specific_cor(df)

    # fol = Cross_Ref_Followers()
    # user_engagements = fol.get_engagements(analysis='tc.topic = "News, Politics & Societal Issues"')
    # user_engagements = fol.merge_following_vs_non_following(user_engagements)
    # fol.compute_ratios_channel_types(user_engagements)
    # per_user = fol.compute_ratios(user_engagements)
    # fol.execute_wilcoxon_test(per_user)

    # user_engagements = fol.get_engagements(analysis='tc.impoliteness >= 5 OR tc.intolerance >= 5')
    # user_engagements = fol.merge_following_vs_non_following(user_engagements)
    # per_user = fol.compute_ratios(user_engagements)
# fol.execute_wilcoxon_test(per_user)




