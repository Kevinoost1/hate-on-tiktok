import pandas as pd
from database import *
import json
from pathlib import Path
import seaborn as sns
import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.stats import f_oneway, pearsonr, shapiro, wilcoxon
from statsmodels.stats.multicomp import MultiComparison
from itertools import combinations
import zstandard as zstd
from exploratory import *
import numpy as np
from scipy.stats import gaussian_kde
from analysis import *

BASE_DIR = Path(__file__).resolve().parent.parent

class Exposure_Graph:
    def __init__(self):
        self.db_path = BASE_DIR/'database/150_db.sqlite3'
        self.db = Database(db_path=self.db_path)

        following = Cross_Ref_Followers

        self.axesticks = 14
        self.fontsize_labels = 34
        self.fontsize_titles = 40
        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d', 
                       'light_blue': '#3a68b0', 'red': '#c43a3a', 'light_green': '#7ab878', 'teal': '#2a8b7a', 'cream': '#f5f0e8',
                       'light_purple': '#d070c8', 'magenta': '#d040b0', 'violet': '#8a2080', 'deep_red': '#b83232', 'light_red': '#d98c8c'}

    def get_exposure_news(self):
        query = f"""
        SELECT news_ratio, user_id
        FROM(
            SELECT AVG(
            CASE WHEN tc.topic = 'News, Politics & Societal Issues' THEN 1
            ELSE 0
        END) AS news_ratio, e.user_id AS user_id
        FROM engagements e
        LEFT JOIN transcript_classification tc ON tc.video_id=e.video_id
        WHERE tc.topic IS NOT NULL AND e.user_id NOT IN (119, 132)
        GROUP BY e.user_id
        )
     
        """
        
        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        return df

    def get_exposure_monthly_news(self):
        query = f""" 
            WITH all_users AS (
                SELECT DISTINCT user_id FROM engagements WHERE user_id NOT IN (119, 132)
            ),
            news_engagements AS (
                SELECT
                    e.user_id,
                    e.video_id,
                    e.date_ymd AS news_date
                FROM engagements e
                INNER JOIN transcript_classification tc ON tc.video_id = e.video_id
                WHERE tc.topic = 'News, Politics & Societal Issues'
                AND e.user_id NOT IN (119, 132)
            ),
            user_timeframes AS (
                SELECT
                    a.user_id,
                    MIN(h.news_date) AS first_news_date,
                    MAX(h.news_date) AS last_news_date,
                    COUNT(h.video_id) AS total_news_videos
                FROM all_users a
                LEFT JOIN news_engagements h ON h.user_id = a.user_id
                GROUP BY a.user_id
            ),
            adjusted_start_dates AS (
                SELECT
                    user_id,
                    first_news_date,
                    last_news_date,
                    total_news_videos,
                    CASE
                        WHEN total_news_videos = 0 THEN NULL
                        ELSE MIN(
                            CASE
                                WHEN first_news_date < '2023-10-01' THEN '2023-10-01'
                                ELSE first_news_date
                            END,
                            last_news_date
                        )
                    END AS start_date
                FROM user_timeframes
            ),
            user_monthly_avgs AS (
                SELECT
                    user_id,
                    total_news_videos,
                    start_date,
                    last_news_date,
                    CASE
                        WHEN total_news_videos = 0 THEN 0
                        ELSE
                            total_news_videos * 1.0 /
                                CASE
                                    WHEN (julianday(last_news_date) - julianday(start_date)) / 30.44 = 0
                                    THEN 1
                                    ELSE (julianday(last_news_date) - julianday(start_date)) / 30.44
                                END
                    END AS monthly_avg_news
                FROM adjusted_start_dates
            )
            SELECT
                user_id,
                total_news_videos,
                start_date,
                last_news_date,
                ROUND(monthly_avg_news, 4) AS monthly_avg_news
            FROM user_monthly_avgs
            ORDER BY monthly_avg_news DESC;        
            """
 
        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        return df


    def get_ratio_following_news(self):
        fol = Cross_Ref_Followers()
        user_engagements = fol.get_engagements(analysis='tc.topic="News, Politics & Societal Issues"')
        user_engagements = fol.merge_following_vs_non_following(user_engagements)
        df = fol.compute_ratios(user_engagements)

        return df

    def get_viewing_durations_news(self):
        data = DataExtract(self.db)
        feed_data = data.request_time_series()
        df = feed_data[0]

        df = df[df['topic'].notna()]
        df = df[df['new_session'] == 0]

        per_user = df.groupby('user_id').apply(lambda x: pd.Series({'news_med_wt': x.loc[x['is_news'] == 1, 'watch_time_s'].median(), 
                                                                   'no_news_med_wt': x.loc[x['is_news'] == 0, 'watch_time_s'].median()}))

        per_user['news_wt_med_diff'] = per_user['news_med_wt'] - per_user['no_news_med_wt']

        return per_user



    def get_exposure_ratio(self, threshold=5) -> pd.DataFrame:
        query = f"""
        SELECT hate_ratio, user_id
        FROM(
            SELECT AVG(
            CASE WHEN tc.intolerance >= {threshold} OR tc.impoliteness >= {threshold} THEN 1
            ELSE 0
        END) AS hate_ratio, e.user_id AS user_id
        FROM engagements e
        LEFT JOIN transcript_classification tc ON tc.video_id=e.video_id
        WHERE tc.topic = "News, Politics & Societal Issues" AND e.user_id NOT IN (119, 132)
        GROUP BY e.user_id
        )
        
        """
        
        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        return df

    def get_exposure_monthly(self, threshold=5) -> pd.DataFrame:
        query = f""" 
            WITH all_users AS (
                SELECT DISTINCT user_id FROM engagements WHERE user_id NOT IN (119, 132)
            ),
            hate_engagements AS (
                SELECT
                    e.user_id,
                    e.video_id,
                    e.date_ymd AS hate_date
                FROM engagements e
                INNER JOIN transcript_classification tc ON tc.video_id = e.video_id
                WHERE tc.intolerance >= 5 OR tc.impoliteness >= 5
                AND tc.topic = "News, Politics & Societal Issues"
                AND e.user_id NOT IN (119, 132)
            ),
            user_timeframes AS (
                SELECT
                    a.user_id,
                    MIN(h.hate_date) AS first_hate_date,
                    MAX(h.hate_date) AS last_hate_date,
                    COUNT(h.video_id) AS total_hate_videos
                FROM all_users a
                LEFT JOIN hate_engagements h ON h.user_id = a.user_id
                GROUP BY a.user_id
            ),
            adjusted_start_dates AS (
                SELECT
                    user_id,
                    first_hate_date,
                    last_hate_date,
                    total_hate_videos,
                    CASE
                        WHEN total_hate_videos = 0 THEN NULL
                        ELSE MIN(
                            CASE
                                WHEN first_hate_date < '2023-10-01' THEN '2023-10-01'
                                ELSE first_hate_date
                            END,
                            last_hate_date
                        )
                    END AS start_date
                FROM user_timeframes
            ),
            user_monthly_avgs AS (
                SELECT
                    user_id,
                    total_hate_videos,
                    start_date,
                    last_hate_date,
                    CASE
                        WHEN total_hate_videos = 0 THEN 0
                        ELSE
                            total_hate_videos * 1.0 /
                                CASE
                                    WHEN (julianday(last_hate_date) - julianday(start_date)) / 30.44 = 0
                                    THEN 1
                                    ELSE (julianday(last_hate_date) - julianday(start_date)) / 30.44
                                END
                    END AS monthly_avg_hate
                FROM adjusted_start_dates
            )
            SELECT
                user_id,
                total_hate_videos,
                start_date,
                last_hate_date,
                ROUND(monthly_avg_hate, 4) AS monthly_avg_hate
            FROM user_monthly_avgs
            ORDER BY monthly_avg_hate DESC;        
            """
 
        with self.db.connect() as db:
            df = pd.read_sql(query, db)

        return df

    def get_ratio_following(self):
        fol = Cross_Ref_Followers()
        user_engagements = fol.get_engagements(analysis='tc.impoliteness >= 5 OR tc.intolerance >= 5')
        user_engagements = fol.merge_following_vs_non_following(user_engagements)
        df = fol.compute_ratios(user_engagements)

        return df

    def get_viewing_durations(self):
        data = DataExtract(self.db)
        feed_data = data.request_time_series()
        df = feed_data[0]

        df = df[df['topic'] == 'News, Politics & Societal Issues']
        df = df[df['new_session'] == 0]

        per_user = df.groupby('user_id').apply(lambda x: pd.Series({'hate_med_wt': x.loc[x['high_hate'] == 1, 'watch_time_s'].median(), 
                                                                   'no_hate_med_wt': x.loc[x['high_hate'] == 0, 'watch_time_s'].median()}))

        per_user['hate_wt_med_diff'] = per_user['hate_med_wt'] - per_user['no_hate_med_wt']

        return per_user


    def _manual_kde_percentage(self, data, bins=30, color=None, ax=None, alpha_fill=0.2, linewidth=2.5, zorder_curve=3, zorder_fill=2):
        """Compute KDE and draw it on the same percentage scale as histplot(stat='percent')."""
        if len(data) < 2 or data.nunique() <= 1:
            return None

        bin_width = (data.max() - data.min()) / bins
        kde = gaussian_kde(data)
        x_grid = np.linspace(data.max() * 0.99, data.max(), 500)

        # Compute the tallest histogram bar (same bins, stat="percent")
        hist_max_count = np.max(np.histogram(data, bins=bins)[0])
        hist_max_percent = hist_max_count / len(data) * 100

        # Bisection: find the bandwidth that makes the KDE peak match the
        # tallest histogram bar.  Narrower bandwidth → spikier KDE → higher peak.
        lo, hi = 0.01, 10.0
        for _ in range(40):
            mid = (lo + hi) / 2.0
            kde.set_bandwidth(mid)
            y_kde = kde(x_grid)
            y_grid = y_kde * 100 * bin_width
            kde_peak = np.max(y_grid)
            ratio = kde_peak / hist_max_percent

            if 0.95 <= ratio <= 1.05:
                break
            elif ratio > 1.05:
                lo = mid      # KDE too tall → need broader bandwidth
            else:
                hi = mid      # KDE too flat → need narrower bandwidth

        # Draw the KDE curve and fill area.
        ax.plot(x_grid, y_grid, color=color, linewidth=linewidth, zorder=zorder_curve)
        ax.fill_between(x_grid, y_grid, color=color, alpha=alpha_fill, zorder=zorder_fill)
        return y_grid


    def create_exposure_news(self) -> plt.Figure:
        """
        Four-panel figure with three percentage histograms/KDE panels and
        a narrow right panel showing the per-user median watching time for
        news vs. non-news videos as a box plot.
        """

        news_exposure_5 = self.get_exposure_news()["news_ratio"].dropna()
        news_monthly_5 = self.get_exposure_monthly_news()["monthly_avg_news"].dropna()
        following_5 = self.get_ratio_following_news()["is_following_ratio"].dropna()

        # Per-user median watching time (news vs. non-news)
        viewing = self.get_viewing_durations_news()
        news_diff = viewing["news_wt_med_diff"].dropna()
        news_med_wt = viewing["news_med_wt"].dropna()
        no_news_med_wt = viewing["no_news_med_wt"].dropna()

        # 3×3 grid — left: 3 panels, middle: violin plot, right: white space
        fig = plt.figure(figsize=(38, 18))
        gs = fig.add_gridspec(
            3, 3,
            width_ratios=[3.25, 1, 0.3],
            hspace=0.5,
            wspace=0.4,
        )

        axes_left = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        ax_right = fig.add_subplot(gs[:, 1])



        # Draw figure first so get_position() returns accurate coordinates
        fig.canvas.draw()

        # Panel center positions for titles and divider
        left_left_x = axes_left[0].get_position().x0
        left_right_x = axes_left[0].get_position().x1
        right_left_x = ax_right.get_position().x0
        right_right_x = ax_right.get_position().x1
        
        # Separate titles for left and right panels
        fig.text(
            (left_left_x + left_right_x) / 2, 0.96,
            "Youths' Exposure to News",
            fontsize=self.fontsize_titles, ha='center', va='top',
        )
        fig.text(
            (right_left_x + right_right_x) / 2 + 0.05, 0.96,
            "Youths' Engagement with News",
            fontsize=self.fontsize_titles, ha='center', va='top',
        )



        panels = [
            (
                axes_left[0], news_exposure_5, "news_ratio",
                self.colors["purple"], "Percentage of Feed Containing News, Politics & Societal Issues (%)", True, 30, False,
            ),
            (
                axes_left[1], news_monthly_5, "monthly_avg_news",
                self.colors["orange"], "Average Number of News Videos per Month", False, 80, False,
            ),
            (
                axes_left[2], following_5, "is_following_ratio",
                self.colors["green"], "Percentage of News from Followed Accounts (%)", True, 30, True,
            ),
        ]

        def plot_percentage_distribution(
            ax, values, color, xlabel, percent_x=False, bins=40, use_median=False
        ):
            values = values[np.isfinite(values)].astype(float)

            if len(values) == 0:
                ax.set_xlabel(xlabel, fontsize=self.fontsize_labels)
                ax.set_ylabel("% of User IDs", fontsize=self.fontsize_labels)
                return

            # Convert ratios to percentage units for display where appropriate.
            scale = 100 if percent_x else 1
            x = values * scale

            # Use fixed, common-width bins for the histogram and KDE scaling.
            if np.min(x) == np.max(x):
                ax.hist(
                    x,
                    bins=[x[0] - 0.5, x[0] + 0.5],
                    weights=np.full(len(x), 100 / len(x)),
                    color=color,
                    alpha=0.25,
                    edgecolor=color,
                )
                ax.axvline(
                    x[0], color=color, linewidth=2.5,
                    label=f"All users = {x[0]:.1f}",
                )
            else:
                edges = np.histogram_bin_edges(x, bins=bins)
                bin_width = edges[1] - edges[0]

                sns.histplot(
                    x=x,
                    bins=edges,
                    stat="percent",
                    color=color,
                    ax=ax,
                    alpha=0.25,
                    edgecolor=None,
                    zorder=1,
                )

                if len(x) >= 2 and np.std(x) > 0:
                    sns.kdeplot(
                        x=x,
                        color=color,
                        ax=ax,
                        linewidth=2.5,
                        fill=False,
                        bw_adjust=0.5,
                        cut=0,
                        zorder=2,
                    )

                    line = ax.lines[-1]
                    line.set_ydata(line.get_ydata() * bin_width * 100)
                    ax.fill_between(
                        line.get_xdata(), line.get_ydata(),
                        color=color, alpha=0.1, zorder=2,
                    )

                stat_x = np.median(x) if use_median else np.mean(x)
                ax.axvline(
                    stat_x,
                    linestyle="--",
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"Median = {stat_x:.2f}" if use_median else f"Mean = {stat_x:.2f}",
                )

            ax.set_xlabel(xlabel, fontsize=self.fontsize_labels)
            ax.set_ylabel("Percentage of\nParticipants (%)", fontsize=self.fontsize_labels)
            ax.tick_params(axis="both", labelsize=self.fontsize_labels)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            # Hide the 0 tick label on the x-axis
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '' if x == 0 else f'{x:.0f}'))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # Set y-ticks including 0 and the max y-value explicitly
            y_ticks = ax.get_yticks()
            y_max = max(y_ticks) if len(y_ticks) > 0 else 0
            ax.set_yticks([0, y_max])

            ax.legend(
                fontsize=self.fontsize_labels * 0.7,
                loc="upper right",
            )

        for ax, values, _, color, xlabel, percent_x, bins, use_median in panels:
            plot_percentage_distribution(
                ax, values, color, xlabel, percent_x=percent_x, bins=bins,
                use_median=use_median
            )

        axes_left[0].set_xlim(0, 9)
        axes_left[0].set_ylim(0)

        axes_left[1].set_xlim(0, 500)
        axes_left[2].set_xlim(0, 13)

        # ---- 4th panel: box plot of per-user median watching time ----
        ax_right.set_title(
            "",
            fontsize=self.fontsize_titles,
        )

        parts = ax_right.boxplot(
            [news_med_wt.values, no_news_med_wt.values],
            positions=[1, 2],
            widths=0.5,
            patch_artist=True,
            showmeans=False,
            showfliers=False,
        )
        for box_color, box in zip([self.colors['red'], self.colors['light_red']], parts["boxes"]):
            box.set_facecolor(box_color)
            box.set_alpha(0.4)
            box.set_edgecolor(box_color)
            box.set_linewidth(2)
        for box_color, line in zip([self.colors['red'], self.colors['light_red']], parts["caps"] + parts["whiskers"]):
            line.set_color(box_color)
            line.set_linewidth(2)
        for box_color, med in zip([self.colors['red'], self.colors['light_red']], parts["medians"]):
            med.set_color(box_color)
            med.set_linewidth(3)

        news_median = news_med_wt.median()
        no_news_median = no_news_med_wt.median()

        legend_news = Line2D([], [], color=self.colors["red"], linewidth=2.5,
                             label=f"News Median = {news_median:.2f}s")
        legend_no_news = Line2D([], [], color=self.colors["light_red"], linewidth=2.5,
                                 label=f"Non-News Median = {no_news_median:.2f}s")

        ax_right.set_ylim(0)

        ax_right.set_ylabel(
            "Median Viewing Time per User (sec)", fontsize=self.fontsize_labels
        )
        ax_right.tick_params(axis="both", labelsize=self.fontsize_labels)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.legend(
            handles=[legend_news, legend_no_news],
            fontsize=self.fontsize_labels * 0.7,
            loc="upper left",
        )

        ax_right.set_xticks([1, 2])
        ax_right.set_xticklabels(
            ["News", "Non-News"], fontsize=self.fontsize_labels * 0.75
        )

        ax_right.text(
            x=1.02, y=1, rotation=0, rotation_mode="anchor",
            s="Median\nViewing Time",
            va="center", ha="left",
            fontsize=self.fontsize_labels * 0.70,
            transform=ax_right.get_yaxis_transform(),
            clip_on=False,
        )

        plt.show()
        return fig


    def create_exposure(self) -> plt.Figure:
        """
        Four-panel figure with three percentage histograms/KDE panels and
        a narrow right panel showing the per-user median watching-time
        difference (hate − non-hate) as a violin plot.
        """

        hate_exposure_5 = self.get_exposure_ratio()["hate_ratio"].dropna()
        hate_monthly_5 = self.get_exposure_monthly()["monthly_avg_hate"].dropna()
        following_5 = self.get_ratio_following()["is_following_ratio"].dropna()

        # Per-user median difference in watching time (hate − non-hate)
        viewing = self.get_viewing_durations()
        hate_diff = viewing["hate_wt_med_diff"].dropna()
        hate_med_wt = viewing["hate_med_wt"].dropna()
        no_hate_med_wt = viewing["no_hate_med_wt"].dropna()

        # 3×3 grid — left: 3 panels, middle: violin plot, right: white space
        fig = plt.figure(figsize=(38, 18))
        gs = fig.add_gridspec(
            3, 3,
            width_ratios=[3.25, 1, 0.3],
            hspace=0.5,
            wspace=0.4,
        )

        axes_left = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        ax_right = fig.add_subplot(gs[:, 1])



        # Draw figure first so get_position() returns accurate coordinates
        fig.canvas.draw()

        # Panel center positions for titles and divider
        left_left_x = axes_left[0].get_position().x0
        left_right_x = axes_left[0].get_position().x1
        right_left_x = ax_right.get_position().x0
        right_right_x = ax_right.get_position().x1
        
        # Separate titles for left and right panels
        fig.text(
            (left_left_x + left_right_x) / 2, 0.96,
            "Youths' Exposure to Hateful News",
            fontsize=self.fontsize_titles, ha='center', va='top',
        )
        fig.text(
            (right_left_x + right_right_x) / 2 + 0.05, 0.96,
            "Youths' Engagement with Hateful News",
            fontsize=self.fontsize_titles, ha='center', va='top',
        )



        panels = [
            (
                axes_left[0], hate_exposure_5, "hate_ratio",
                self.colors["purple"], "Percentage of News Feed Containing Hate (%)", True, 30, False,
            ),
            (
                axes_left[1], hate_monthly_5, "monthly_avg_hate",
                self.colors["orange"], "Average Number of Hate Videos per Month", False, 80, False,
            ),
            (
                axes_left[2], following_5, "is_following_ratio",
                self.colors["green"], "Percentage of Hate from Followed Accounts (%)", True, 30, True,
            ),
        ]

        def plot_percentage_distribution(
            ax, values, color, xlabel, percent_x=False, bins=40, use_median=False
        ):
            values = values[np.isfinite(values)].astype(float)

            if len(values) == 0:
                ax.set_xlabel(xlabel, fontsize=self.fontsize_labels)
                ax.set_ylabel("% of User IDs", fontsize=self.fontsize_labels)
                return

            # Convert ratios to percentage units for display where appropriate.
            scale = 100 if percent_x else 1
            x = values * scale

            # Use fixed, common-width bins for the histogram and KDE scaling.
            if np.min(x) == np.max(x):
                ax.hist(
                    x,
                    bins=[x[0] - 0.5, x[0] + 0.5],
                    weights=np.full(len(x), 100 / len(x)),
                    color=color,
                    alpha=0.25,
                    edgecolor=color,
                )
                ax.axvline(
                    x[0], color=color, linewidth=2.5,
                    label=f"All users = {x[0]:.1f}",
                )
            else:
                edges = np.histogram_bin_edges(x, bins=bins)
                bin_width = edges[1] - edges[0]

                sns.histplot(
                    x=x,
                    bins=edges,
                    stat="percent",
                    color=color,
                    ax=ax,
                    alpha=0.25,
                    edgecolor=None,
                    zorder=1,
                )

                if len(x) >= 2 and np.std(x) > 0:
                    sns.kdeplot(
                        x=x,
                        color=color,
                        ax=ax,
                        linewidth=2.5,
                        fill=False,
                        bw_adjust=0.5,
                        cut=0,
                        zorder=2,
                    )

                    line = ax.lines[-1]
                    line.set_ydata(line.get_ydata() * bin_width * 100)
                    ax.fill_between(
                        line.get_xdata(), line.get_ydata(),
                        color=color, alpha=0.1, zorder=2,
                    )

                stat_x = np.median(x) if use_median else np.mean(x)
                ax.axvline(
                    stat_x,
                    linestyle="--",
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"Median = {stat_x:.2f}" if use_median else f"Mean = {stat_x:.2f}",
                )

            ax.set_xlabel(xlabel, fontsize=self.fontsize_labels)
            ax.set_ylabel("PercPercentagee of\nParticipants (%)", fontsize=self.fontsize_labels)
            ax.tick_params(axis="both", labelsize=self.fontsize_labels)
            ax.xaxis.set_major_locator(MaxNLocator(5))
            # Hide the 0 tick label on the x-axis
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: '' if x == 0 else f'{x:.0f}'))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            # Set y-ticks including 0 and the max y-value explicitly
            y_ticks = ax.get_yticks()
            y_max = max(y_ticks) if len(y_ticks) > 0 else 0
            ax.set_yticks([0, y_max])

            ax.legend(
                fontsize=self.fontsize_labels * 0.7,
                loc="upper right",
            )

        for ax, values, _, color, xlabel, percent_x, bins, use_median in panels:
            plot_percentage_distribution(
                ax, values, color, xlabel, percent_x=percent_x, bins=bins,
                use_median=use_median
            )

        axes_left[0].set_xlim(0, 9)
        axes_left[0].set_ylim(0)

        axes_left[1].set_xlim(0, 25)
        axes_left[2].set_xlim(0, 13)

        # ---- 4th panel: box plot of per-user median watching time ----
        ax_right.set_title(
            "",
            fontsize=self.fontsize_titles,
        )

        parts = ax_right.boxplot(
            [hate_med_wt.values, no_hate_med_wt.values],
            positions=[1, 2],
            widths=0.5,
            patch_artist=True,
            showmeans=False,
            showfliers=False,
        )

        for box_color, box in zip([self.colors['red'], self.colors['light_red']], parts["boxes"]):
            box.set_facecolor(box_color)
            box.set_alpha(0.4)
            box.set_edgecolor(box_color)
            box.set_linewidth(2)
        for box_color, line in zip([self.colors['red'], self.colors['light_red']], parts["caps"] + parts["whiskers"]):
            line.set_color(box_color)
            line.set_linewidth(2)
        for box_color, med in zip([self.colors['red'], self.colors['light_red']], parts["medians"]):
            med.set_color(box_color)
            med.set_linewidth(3)

        hate_median = hate_med_wt.median()
        no_hate_median = no_hate_med_wt.median()

        legend_hate = Line2D([], [], color=self.colors["red"], linewidth=2.5,
                             label=f"Hateful Median = {hate_median:.2f}s")
        legend_no_hate = Line2D([], [], color=self.colors["light_red"], linewidth=2.5,
                                 label=f"Non-Hateful Median = {no_hate_median:.2f}s")

        ax_right.set_ylim(0)

        ax_right.set_ylabel(
            "Median Viewing Time per User (sec)", fontsize=self.fontsize_labels
        )
        ax_right.tick_params(axis="both", labelsize=self.fontsize_labels)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["right"].set_visible(False)
        ax_right.legend(
            handles=[legend_hate, legend_no_hate],
            fontsize=self.fontsize_labels * 0.7,
            loc="upper left",
        )

        ax_right.set_xticks([1, 2])
        ax_right.set_xticklabels(
            ["Hateful\nNews", "Non-Hateful\nNews"], fontsize=self.fontsize_labels * 0.75
        )

        ax_right.text(
            x=1.02, y=1, rotation=0, rotation_mode="anchor",
            s="Median\nViewing Time",
            va="center", ha="left",
            fontsize=self.fontsize_labels * 0.70,
            transform=ax_right.get_yaxis_transform(),
            clip_on=False,
        )

        plt.show()
        return fig
    


    def notes(self):
        thresholds = [3, 4, 5, 6]
        ratio_proportions = []
        monthly_proportions = []

        for threshold in thresholds:
            hate_ratio = self.get_exposure_ratio(threshold=threshold)
            ratio_proportions.append(hate_ratio.mean())

            hate_monthly = self.get_exposure_monthly(threshold=threshold)
            monthly_proportions.append(hate_monthly)


    def export_png(self,  *figs: tuple[plt.Figure], custom_label='') -> None:
        for i, fig in enumerate(figs):
            fig.savefig(f'plots/{custom_label}_{i}.png', dpi=300)


    def export_tiff(self, *figs: tuple[plt.Figure], custom_label='exposure') -> None:
        for i, fig in enumerate(figs):
            fig.savefig(f'plots/{custom_label}_{i}.tiff', format='tiff', dpi=600)


if __name__=='__main__':
    graph = Exposure_Graph()
    exposure_plot = graph.create_exposure()
    # exposure_plot_news = graph.create_exposure_news()

    # graph.export_png(exposure_plot, exposure_plot_news)
    # graph.export_tiff(exposure_plot)
