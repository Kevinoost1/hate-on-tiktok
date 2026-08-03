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

BASE_DIR = Path(__file__).resolve().parent.parent

class Hate_Speech_Target:
    def __init__(self):
        self.keywords = self.load_keywords()

        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}
        self.fontsize_labels: int = 32
        self.fontsize_titles: int = 36
        self.axesticks: int = 28

        self.category_groups = {
            # Dutch politics
            "PVV": ("Dutch Politics", "right"),
            "FVD": ("Dutch Politics", "right"),
            "VVD": ("Dutch Politics", "center-right"),
            "BBB": ("Dutch Politics", "right"),
            "D66": ("Dutch Politics", "center-left"),
            "SP": ("Dutch Politics", "left"),
            "Groen Links PVDA": ("Dutch Politics", "left"),
            "CDA": ("Dutch Politics", "center-right"),
            "Denk": ("Dutch Politics", "left"),

            "Geert Wilders": ("Dutch Politics", "right"),
            "Thierry Baudet": ("Dutch Politics", "right"),
            "Mark Rutte": ("Dutch Politics", "center-right"),
            "Frans Timmermans": ("Dutch Politics", "left"),
            "Jesse Klaver": ("Dutch Politics", "left"),
            "Rob Jetten": ("Dutch Politics", "center-left"),

            # US politics
            "Donald Trump": ("US Politics", "right"),
            "MAGA": ("US Politics", "right"),
            "Republicans": ("US Politics", "right"),
            "Joe Biden": ("US Politics", "left"),
            "Kamala Harris": ("US Politics", "left"),
            "Democrats": ("US Politics", "left"),
            "Charlie Kirk": ("US Politics", "right"),
            "J.D. Vance": ("US Politics", "right"),
            "Barack Obama": ("US Politics", "left"),

            # Religion
            "Muslims": ("Religion", None),
            "Christians": ("Religion", None),
            "Jewish People": ("Religion", None),
            "Hindus": ("Religion", None),

            # Refugees
            "Refugees & Migrants": ("Refugees & Migrants", None),

            # Active Conflict
            "Gaza": ("Active Conflict", None),
            "Ukraine": ("Active Conflict", None),
            "Vladimir Putin": ("Active Conflict", None),
            "Zelensky": ("Active Conflict", None),

            # Minority Groups
            "Black People": ("Minority Groups", None),
            "Moroccans": ("Minority Groups", None),
            "Turks": ("Minority Groups", None), 
            "Hispanic People": ("Minority Groups", None),
            "Disabled & Neurodivergents": ("Minority Groups", None),
            "Africans": ("Minority Groups", None),
            "Arab People": ("Minority Groups", None),
            "Chinese": ("Minority Groups", None),
            "Asian People": ("Minority Groups", None),

            # LGBTQ
            "LGBTQ+": ("LGBTQ+ & Gender", None),
            "Gay People": ("LGBTQ+ & Gender", None),
            "(Trans)gender": ("LGBTQ+ & Gender", None),
        }

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

        return freq, freq_df

    def match_words(self, text, keywords) -> Iterator[str]:
        for category, words in self.keywords.items():
            for word in words:
                if word in text:
                    yield category
                    break


    def combine_transcript_desc(self, transcript, desc) -> str:
        return transcript + " " + desc.replace("#", "").strip()
    
    def load_keywords(self) -> dict[str, int]:
        with open(BASE_DIR/'src/keywords.json', 'r') as f:
            keywords = json.load(f)

        keyword_map = {}
        for broad_category, specifics in keywords['keywords'].items():
            for specific, keywords in specifics.items():
                keyword_map[specific] = [w for w in keywords]

        return keyword_map

    def get_hate_speech(self) -> pd.DataFrame:
        db = Database(db_path='/home/kevin/Documents/phd/Project-Data-Donation/database/db.sqlite3')
        return db.retrieve_data(
            tableName = 'transcript_classification tc LEFT JOIN videos v ON tc.video_id=v.id',
            colNames = ['tc.id', 'tc.translation', 'v.desc'],
            restrictions = 'tc.impoliteness > 5'
        ) 

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
        self.db = Database()
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
            WHERE u.id != 119
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


        titles = {
            "age": "Age",
            "future_anxiety_total": "Future Anxiety",
            "sr_hate_speech": "Self-reported Hate Speech",
            "affective_pol": "Perceived Affective Polarization",
        }

        fig, axes = plt.subplots(2, 2, figsize=(20, 20), sharey=True)
        axes = axes.flatten()

        fig.suptitle(
            "Relationships with Encountered Hate in Feed",
            fontsize=self.fontsize_titles,
            y=0.80
        )

        # Transparent orange for CI
        orange_alpha = to_rgba(self.colors["orange"], alpha=0.25)

        likert_vars = {
            "future_anxiety_total",
            "sr_hate_speech",
            "affective_pol",
            "age"
        }

        for i, (ax, var) in enumerate(zip(axes, vars)):
            local_df = df[["high_hate_ratio", var]].dropna()

            sns.regplot(
                data=local_df,
                x=var,
                y="high_hate_ratio",
                ax=ax,
                scatter_kws={
                    "color": self.colors["purple"],
                    "alpha": 0.6,
                    "s": 40,
                },
                line_kws={
                    "color": self.colors["orange"],
                    "lw": 3,
                },
                color=orange_alpha,
            )

            # Move only x-axis label to the top (keep ticks at bottom)
            ax.xaxis.set_label_position("top")

            ax.set_xlabel(
                titles[var],
                fontsize=self.fontsize_labels,
                labelpad=15,
            )

            # Only add y-axis label to the left column
            if i in [0, 2]:
                ax.set_ylabel(
                    "High hate ratio",
                    fontsize=self.fontsize_labels,
                )
            else:
                ax.set_ylabel("")

            ax.spines["bottom"].set_visible(True)
            ax.spines["bottom"].set_linewidth(0.8)

            ax.tick_params(
                axis="both",
                labelsize=self.axesticks,
                bottom=True,
                length=4,
                width=0.8,
                color="black"
            )

            # Integer ticks for Likert variables only
            if var in likert_vars:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            # Add Pearson r and significance
            result_row = results.loc[results["var"] == var].iloc[0]

            r = result_row["cor"]
            p = result_row["corrected_p"]

            if p < 0.001:
                stars = "***"
            elif p < 0.01:
                stars = "**"
            elif p < 0.05:
                stars = "*"
            else:
                stars = ""

            ax.text(
                0.05,
                0.90,
                rf"$r$ = {r:.2f}{stars}",
                transform=ax.transAxes,
                fontsize=self.axesticks,
                verticalalignment="top",
            )

        sns.despine(bottom=False, left=False)

        # Adjust spacing: move top row closer to title and increase row separation
        plt.subplots_adjust(
            top=0.74,
            hspace=0.35
        )

        fig.savefig(
            "plots/cor.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.25
        )

        plt.show()     



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
    hate_target = Hate_Speech_Target()
    cor = Correlation_Matrix()

    freq, freq_df = hate_target.construct_freq()
    plot = hate_target.create_frequency_graph2(freq)
    print(freq)
    

    df = cor.get_data()
    # cor_matrix = cor.get_cor_matrix(df, columns_to_drop=['id', 'name'])
    # print(cor_matrix)
    # cor_matrix.to_csv('cor_matrix.csv')
    # cor.plot_cor_matrix(cor_matrix)

    specific_cors = cor.specific_cor(df)
    print(specific_cors.head(10))


