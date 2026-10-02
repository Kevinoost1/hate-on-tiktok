import pandas as pd
from database import *
import json
import re
from pathlib import Path
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from matplotlib.transforms import blended_transform_factory

BASE_DIR = Path(__file__).resolve().parent.parent

class Hate_Speech_Target:
    def __init__(self):
        self.keywords = self.load_keywords()

        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}
        self.fontsize_labels: int = 32
        self.fontsize_titles: int = 36
        self.axesticks: int = 28

    def construct_freq(self) -> pd.DataFrame:
        df = self.get_hate_speech() 
        df['text'] = df.apply(lambda x: self.combine_transcript_desc(x['translation'], x['desc']), axis=1)

        freq = {k: 0 for k in self.keywords.keys()}
        freq_df = df.copy().assign(**freq)

        for i, text in enumerate(df['text'].str.lower()):
            checked_categories = set(self.match_words(text, self.keywords))

            for cat in checked_categories:
                freq[cat] += 1
                freq_df.at[i, cat] += 1

        freq, freq_df = self.subdivide_by_channel(freq_df)
        ratio_freq = self.construct_ratio(freq)

        return ratio_freq, freq_df

    def match_words(self, text, keywords) -> Iterator[str]:
        for category, words in self.keywords.items():
            for word in words:
                if word in text:
                    yield category
                    break


    def combine_transcript_desc(self, transcript, desc) -> str:
        return transcript + " " + desc.replace("#", "").strip()
    
    def load_keywords(self) -> dict[str, int]:
        with open(BASE_DIR/'src/keywords2.json', 'r') as f:
            keywords = json.load(f)

        keyword_map = {}
        for broad_category, keyword in keywords['keywords'].items():
            keyword_map[broad_category] = [w for w in keyword]

        return keyword_map

    def get_hate_speech(self) -> pd.DataFrame:
        db = Database(db_path=BASE_DIR/'database/150_db.sqlite3')
        return db.retrieve_data(
            tableName = 'transcript_classification tc LEFT JOIN videos v ON tc.video_id=v.id',
            colNames = ['tc.id', 'tc.video_id', 'tc.translation', 'v.desc', 'tc.original_language'],
            restrictions = 'tc.impoliteness >= 5 OR tc.intolerance >= 5'
        ) 

    def subdivide_by_channel(self, freq_df):
        categories = ['Minority Groups', 'Active Conflict', 'Religion', 'LGBTQ+ & Gender', 'Refugees & Migrants', 'Other']
        for cat in categories:
            legacy_col = f'{cat} Legacy'
            alternative_col = f'{cat} Alternative'
            niche_col = f'{cat} Niche'

            freq_df[legacy_col] = ((freq_df['official'] == 1) & (freq_df[cat] == 1)).astype(int)
            freq_df[alternative_col] = ((freq_df['official'] == 2) & (freq_df[cat] == 1)).astype(int)
            freq_df[niche_col] = ((freq_df['official'] == 3) & (freq_df[cat] == 1)).astype(int)

            freq_df.drop(columns=cat, inplace=True)
            freq = {col: int(freq_df[col].sum()) for col in freq_df.columns if col not in [
                    'id', 'video_id', 'original_language', 'translation', 'desc', 'text']}

            return freq, freq_df




    def subdivide_by_language(self, freq_df):
        categories = ['Minority Groups', 'Active Conflict', 'Religion', 'LGBTQ+ & Gender', 'Refugees & Migrants', 'Other']
        for cat in categories:
            dutch_col = f'{cat} Dutch'
            us_col = f'{cat} US'

            freq_df[dutch_col] = ((freq_df['original_language'] == 18) & (freq_df[cat] == 1)).astype(int)
            freq_df[us_col] = ((freq_df['original_language'] == 2) & (freq_df[cat] == 1)).astype(int)

            freq_df.drop(columns=cat, inplace=True)

        freq = {col: int(freq_df[col].sum()) for col in freq_df.columns if col not in [
                'id', 'video_id', 'original_language', 'translation', 'desc', 'text']}

        return freq, freq_df

    def construct_ratio(self, freq):
        ratio = {}

        dutch = sum(v for k, v in freq.items() if k.endswith('Dutch'))
        us = sum(v for k, v in freq.items() if k.endswith('US'))
        total = dutch + us

        ratio['Dutch'] = dutch / total if total else 0
        ratio['US'] = us / total if total else 0

        for key, value in freq.items():
            if key.endswith('Dutch'):
                ratio[key] = value / dutch if dutch else 0
            elif key.endswith('US'):
                ratio[key] = value / us if us else 0

        return ratio

    def test_graph(self, ratio_dict):

        #
        # Language colors
        #
        group_colors = {
            "Dutch": '#ec8e3a',
            "US": "#001157"
        }

        #
        # Category order
        #
        categories = [
            "Politics",
            "Minority Groups",
            "Active Conflict",
            "Religion",
            "LGBTQ+ & Gender",
            "Refugees & Migrants",
            "Other"
        ]

        #
        # Construct dataframe
        #
        rows = []

        for language in ["Dutch", "US"]:

            for cat in categories:

                if cat == "Politics":
                    key = f"Politics {language}"
                else:
                    key = f"{cat} {language}"

                rows.append({
                    "label": cat,
                    "ratio": ratio_dict[key],
                    "language": language
                })

        df = pd.DataFrame(rows)

        #
        # Y positions
        #
        y_positions = []
        current_y = 0
        language_positions = {}

        for language in ["Dutch", "US"]:

            start = current_y

            for _ in range(len(categories)):
                y_positions.append(current_y)
                current_y += 1

            end = current_y - 1
            language_positions[language] = (start, end)

            # whitespace between Dutch and US
            current_y += 1


        fig, ax = plt.subplots(figsize=(40, 26))


        #
        # Background language bars
        #
        for language, (start, end) in language_positions.items():

            midpoint = (start + end) / 2

            ax.barh(
                midpoint,
                ratio_dict[language],
                height=(end - start) + 1.15,
                color=group_colors[language],
                alpha=0.14,
                zorder=0
            )


        #
        # Individual category bars
        #
        for y, (_, row) in zip(y_positions, df.iterrows()):

            ax.barh(
                y,
                row["ratio"],
                color=group_colors[row["language"]],
                height=0.60,
                zorder=2
            )


        #
        # Ratio labels for categories
        #
        for y, (_, row) in zip(y_positions, df.iterrows()):

            ax.text(
                row["ratio"] + 0.01,
                y,
                f"{row['ratio']:.2f}",
                fontsize=self.axesticks,
                va="center"
            )

        label_transform = blended_transform_factory(
            ax.transAxes,   # x in axes coordinates (0-1)
            ax.transData    # y in data coordinates
        )
        #
        # Language section labels
        #
        for language, (start, end) in language_positions.items():

            midpoint = (start + end) / 2

            ax.text(
                0.7,
                midpoint,
                language,
                transform=label_transform,
                ha="left",
                va="center",
                fontsize=self.axesticks,
                fontweight="bold",
                clip_on=False
            )

            ax.text(
                0.81,
                midpoint,
                f"{ratio_dict[language]:.2f}",
                transform=label_transform,
                ha="left",
                va="center",
                fontsize=self.axesticks,
                fontweight="normal",
                clip_on=False
            )
        #
        # Y labels
        #
        ax.set_yticks(y_positions)
        ax.set_yticklabels(
            df["label"],
            fontsize=self.axesticks
        )

        ax.invert_yaxis()


        #
        # Titles
        #
        ax.set_xlabel(
            "\nRatio of Subtopic Appearance",
            fontsize=self.fontsize_labels
        )

        ax.set_title(
            "\nSubtopics in Hate Speech\nby National Context",
            fontsize=self.fontsize_titles,
            pad=18
        )


        ax.tick_params(
            axis="x",
            labelsize=self.axesticks
        )


        sns.despine(left=True)


        #
        # Layout
        #
        plt.subplots_adjust(
            left=0.28,
            right=0.95,
            top=0.88,
            bottom=0.15
        )


        fig.savefig(
            "plots/hate_target_language.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.25
        )


        plt.show()

        return fig

















    def create_frequency_graph(self, freq_dict: Dict[str, int], top_cat=40) -> plt.Figure():
        freq = pd.Series(freq_dict).sort_values(ascending=False)
        top = freq.head(top_cat)

        fig, ax = plt.subplots(figsize=(30, 20))
        sns.barplot(x=top.values, y=top.index, ax=ax, color=self.colors['purple'])
        ax.set_xlabel('Number of Video Appearances', fontsize=self.fontsize_labels)
        ax.set_ylabel('Groups and Political Actors', fontsize=self.fontsize_labels)

        ax.set_title(f'Top {top_cat} Groups and Political Actors\nin High Intolerance Videos', fontsize=self.fontsize_titles)
        ax.tick_params(axis='both', labelsize=self.axesticks)

        sns.despine()
        plt.show()

        fig.savefig(f'plots/hate_target.png', dpi=300)

        return fig


    def create_frequency_graph2(self, freq_dict: Dict[str, int], top_cat=40):

        df = pd.DataFrame(
            {
                "category": list(freq_dict.keys()),
                "frequency": list(freq_dict.values())
            }
        )


        # Add metadata
        df["group"] = df["category"].apply(
            lambda x: self.category_groups.get(
                x,
                ("Other", None)
            )[0]
        )

        df["position"] = df["category"].apply(
            lambda x: self.category_groups.get(
                x,
                ("Other", None)
            )[1]
        )


        # Keep top categories
        df = (
            df.sort_values(
                "frequency",
                ascending=False
            )
            .head(top_cat)
        )


        # Desired top-to-bottom order
        group_order = [
            "Dutch Politics",
            "US Politics",
            "Minority Groups",
            "Active Conflict",
            "Religion",
            "LGBTQ+ & Gender",
            "Refugees & Migrants",
            "Other"
        ]


        df["group"] = pd.Categorical(
            df["group"],
            categories=group_order,
            ordered=True
        )


        df = df.sort_values(
            ["group", "frequency"],
            ascending=[True, False]
        )


        #
        # Broad category colors
        #
        group_colors = {

            "Dutch Politics": "#244A7C",
            "US Politics": "#244A7C",
            "Minority Groups": "#3F7C7A",
            "Active Conflict": "#001157",
            "Religion": "#568C55",
            "LGBTQ+ & Gender": "#B03AA3",
            "Refugees & Migrants": "#8C4A8D",
            "Other": "#8A8A8A"

        }


        #
        # Political orientation colors
        #
        political_gradient = {

            "left": "#1F4E9E",
            "center-left": "#7FA8D8",
            "center": "#808080",
            "center-right": "#D98282",
            "right": "#A00000"

        }


        #
        # Individual bar colors
        #
        colors = []

        for _, row in df.iterrows():

            if row["group"] in [
                "Dutch Politics",
                "US Politics"
            ]:

                colors.append(
                    political_gradient.get(
                        row["position"],
                        group_colors[row["group"]]
                    )
                )

            else:

                colors.append(
                    group_colors[row["group"]]
                )



        fig, ax = plt.subplots(
            figsize=(40, 26)
        )


        #
        # Group totals
        #
        group_totals = (
            df.groupby(
                "group",
                observed=True
            )["frequency"]
            .sum()
        )


        group_counts = (
            df.groupby(
                "group",
                observed=True
            )
            .size()
        )


        #
        # Create y coordinates with whitespace after politics
        #
        y_positions = []
        group_positions = {}

        current_y = 0

        for group, count in group_counts.items():

            start = current_y

            for _ in range(count):
                y_positions.append(current_y)
                current_y += 1


            end = current_y - 1

            group_positions[group] = (
                start,
                end
            )


            # whitespace after US Politics
            if group == "US Politics":
                current_y += 1



        #
        # Background group total areas
        #
        for group, (start, end) in group_positions.items():

            midpoint = (start + end) / 2

            ax.barh(
                midpoint,
                group_totals[group],
                height=(end - start) + 0.35,
                color=group_colors[group],
                alpha=0.14,
                zorder=0
            )



        #
        # Individual category bars
        #
        ax.barh(
            y_positions,
            df["frequency"],
            color=colors,
            height=0.6,
            zorder=2
        )


        # Put first category at the top
        ax.invert_yaxis()



        #
        # Remove individual category labels
        #
        ax.set_yticks([])



        #
        # Add group labels
        #
        for group, (start, end) in group_positions.items():

            midpoint = (start + end) / 2

            ax.text(
                -0.015,
                midpoint,
                group,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=self.axesticks,
                fontweight="bold",
                clip_on=False
            )



        #
        # Add group totals
        #
        max_frequency = df["frequency"].max()

        for group, (start, end) in group_positions.items():

            midpoint = (start + end) / 2

            ax.text(
                group_totals[group] + max_frequency * 0.03,
                midpoint,
                f"Σ={int(group_totals[group])}",
                fontsize=self.axesticks,
                fontweight="bold",
                va="center"
            )



        #
        # Political legend
        #
        legend_elements = [

            Patch(
                facecolor=political_gradient["left"],
                label="Left"
            ),

            Patch(
                facecolor=political_gradient["center-left"],
                label="Center-left"
            ),

            Patch(
                facecolor=political_gradient["center"],
                label="Center"
            ),

            Patch(
                facecolor=political_gradient["center-right"],
                label="Center-right"
            ),

            Patch(
                facecolor=political_gradient["right"],
                label="Right"
            )

        ]


        ax.legend(
            handles=legend_elements,
            title="Political orientation",
            loc="upper right",
            fontsize=22,
            title_fontsize=26,
            frameon=True,
            borderpad=0.5,
            labelspacing=0.35,
            handlelength=1.3
        )



        #
        # Labels and formatting
        #
        ax.set_xlabel(
            "\nNumber of Video Appearances",
            fontsize=self.fontsize_labels
        )


        ax.set_title(
            "\nGroups and Political Actors\nin High Hate Videos",
            fontsize=self.fontsize_titles,
            pad=18
        )


        ax.tick_params(
            axis="both",
            labelsize=self.axesticks
        )


        sns.despine(
            left=True
        )


        #
        # Layout
        #
        plt.subplots_adjust(
            left=0.20,
            right=0.95,
            top=0.85,
            bottom=0.15
        )


        fig.savefig(
            "plots/hate_target.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.25
        )


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
        q =  self.db.retrieve_data(
            tableName='questionnaires', 
            colNames=['id', 'age', 'political_preference', 'dimi1_compulsion1', 'future_anxiety_1', 'future_anxiety_2', 'future_anxiety_3',
                      'future_anxiety_4', 'future_anxiety_5', 'sr_hate_speech', 'affective_pol', 'future_anxiety_total', 'dimi_compulsion_total', 
                      'dimi_choice_total', 'boredom_total', 'skills_self_control_total', 'skills_misinformation_total',
                        'dimi2_compulsion2', 'dimi3_compulsion3', 'dimi4_choice1','dimi5_choice2', 'dimi6_choice3', 'skills1_self_control1',
                        'skills2_self_control2', 'skills3_self_control3', 'skills4_misinformation1', 'skills5_misinformation2',
                        'skills6_misinformation3', 'boredom1', 'boredom2', 'boredom3', 'news_1', 'news_3', 'news_4_1',
                        'news_4_2', 'ipsos_statements_1', 'ipsos_statements_2', 'ipsos_statements_3', 'ipsos_statements_4',
                        'ipsos_statements_5', 'ipsos_statements_6', 'ipsos_statements_7', 'ipsos_statements_8', 'ipsos_statements_1_opinion_extreme', 
                        'ipsos_statements_2_opinion_extreme', 'ipsos_statements_3_opinion_extreme', 'ipsos_statements_4_opinion_extreme',
                        'ipsos_statements_5_opinion_extreme', 'ipsos_statements_6_opinion_extreme', 'ipsos_statements_7_opinion_extreme', 'ipsos_statements_8_opinion_extreme'], 
        restrictions='id != 119') # participant 119 was excluded due to being to old (age 30)

        official, unofficial, ordering = self.load_channels()
        official_sql = "(" + ",".join(f"'{x}'" for x in official) + ")"
        unofficial_sql = "(" + ",".join(f"'{x}'" for x in unofficial) + ")"

        data_donation_query = f""" SELECT u.id, u.name, AVG(
                    CASE
                        WHEN tc.topic = 'News, Politics & Societal Issues'
                        THEN 1.0
                        ELSE 0.0
                    END) AS news_ratio, AVG(tc.intolerance) AS avg_intolerance, 
                AVG(
                CASE
                    WHEN tc.topic != 'News, Politics & Societal Issues'
                    THEN NULL
                    WHEN tc.impoliteness >= 5 OR tc.intolerance >= 5
                    THEN 1.0
                    ELSE 0.0
                END) AS high_hate_ratio,
                AVG (
                CASE
                    WHEN tc.topic != 'News, Politics & Societal Issues' 
                    THEN NULL
                    WHEN a.unique_id IN {official_sql} THEN 1.0
                    ELSE 0.0
                END) AS official_ratio, 
                AVG(
                CASE
                    WHEN tc.topic != 'News, Politics & Societal Issues'
                    THEN NULL
                    WHEN a.unique_id IN {unofficial_sql} THEN 1.0
                    ELSE 0.0
                END) AS unofficial_ratio, 
                AVG(
                CASE
                    WHEN tc.topic != 'News, Politics & Societal Issues'
                    THEN NULL
                    WHEN a.unique_id NOT IN {official_sql} AND a.unique_id NOT IN {unofficial_sql} THEN 1.0
                    ELSE 0.0
                END) AS unknown_ratio
            FROM users u
            LEFT JOIN engagements e ON e.user_id = u.id
            LEFT JOIN videos v ON v.id=e.video_id
            LEFT JOIN authors a ON a.id=v.author_id
            RIGHT JOIN transcript_classification tc ON tc.video_id = e.video_id
            WHERE u.id NOT IN (119, 132) AND u.id > 50
            GROUP BY u.id, u.name
            ORDER BY u.id ASC;
            """ 
        with self.db.connect() as db:
            data_donation = pd.read_sql(data_donation_query, db)

        print('mean official', data_donation['official_ratio'].mean())
        print('sd official', data_donation['official_ratio'].std())
        print('max official', data_donation['official_ratio'].max())
        print('max official', data_donation['official_ratio'].min())

        print('mean unofficial', data_donation['unofficial_ratio'].mean())
        print('sd unofficial', data_donation['unofficial_ratio'].std())
        print('max unofficial', data_donation['unofficial_ratio'].max())
        print('max unofficial', data_donation['unofficial_ratio'].min())

        print('mean unknown_ratio', data_donation['unknown_ratio'].mean())
        print('sd unknown_ratio', data_donation['unknown_ratio'].std())
        print('max unknown_ratio', data_donation['unknown_ratio'].max())
        print('max unknown_ratio', data_donation['unknown_ratio'].min())

        print('mean high_hate_ratio', data_donation['high_hate_ratio'].mean())
        print('sd high_hate_ratio', data_donation['high_hate_ratio'].std())
        print('min high_hate_ratio', data_donation['high_hate_ratio'].min())
        print('max high_hate_ratio', data_donation['high_hate_ratio'].max())

        return q.merge(data_donation, on='id', how='left')

    def get_cor_matrix(self, df: pd.DataFrame, columns_to_drop: list[str]=[]) -> pd.DataFrame:
        df = df.drop(columns=columns_to_drop) 
        return df.corr()

    def specific_cor(self, df: pd.DataFrame):
        vars = ["age", "future_anxiety_total", "sr_hate_speech", "affective_pol",]
        
        results = []
        for var in vars:
            local_df = df[['high_hate_ratio', var]].dropna()
            r, p = pearsonr(local_df['high_hate_ratio'], local_df[var])

            results.append({'var': var, 'cor': r, 'p': p, 'n': len(local_df)})
        results = pd.DataFrame(results)

        reject, cor_p, _, _= multipletests(results['p'], method='holm')
        results['significant_holm'] = reject
        results['corrected_p'] = cor_p

        return results

    def plot_cor_matrix(self, cor_matrix: pd.DataFrame) -> pd.DataFrame:
        plt.figure(figsize=(8, 8))
        sns.heatmap(cor_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidth=0.5)
        plt.show()

    @staticmethod
    def load_channels(channelDefinitionFileName: str=BASE_DIR / 'src' / 'channel_def.json') -> (list[str], list[str], list[str]):
        with open(channelDefinitionFileName, 'r') as f:
            channels = json.load(f)
        
        return channels['official'], channels['unofficial'], channels['ordering']


if __name__=='__main__':
    # hate_target = Hate_Speech_Target()
    cor = Correlation_Matrix()

    # freq, freq_df = hate_target.construct_freq()
    # plot = hate_target.test_graph(freq)
    # print(freq)
    

    df = cor.get_data()
    # cor_matrix = cor.get_cor_matrix(df, columns_to_drop=['id', 'name'])
    # print(cor_matrix)
    # cor_matrix.to_csv('cor_matrix.csv')
    # cor.plot_cor_matrix(cor_matrix)

    # specific_cors = cor.specific_cor(df)
    # print(specific_cors.head(10))


