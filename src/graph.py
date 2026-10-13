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
import zstandard as zstd
from exploratory import *
import numpy as np
from scipy.stats import gaussian_kde

BASE_DIR = Path(__file__).resolve().parent.parent

class Exposure_Graph:
    def __init__(self):
        self.db_path = BASE_DIR/'database/150_db.sqlite3'
        self.db = Database(db_path=self.db_path)

        following = Cross_Ref_Followers

        self.axesticks = 14
        self.fontsize_labels = 34
        self.fontsize_titles = 40
        self.colors = {'dark_blue': '#001157', 'orange': '#ec8e3a', 'green': '#4F8B4C', 'purple': '#b03aa3', 'brown': '#d67f2d'}

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

    def create_exposure(self) -> plt.Figure:
        """
        Three-panel figure with percentage histograms and KDE curves.

        Histogram: percentage of user_ids in each bin.
        KDE: estimated percentage of user_ids per equivalent bin width.
        Both use the same y-axis interpretation.
        """

        hate_exposure_5 = self.get_exposure_ratio()["hate_ratio"].dropna()
        hate_monthly_5 = self.get_exposure_monthly()["monthly_avg_hate"].dropna()
        following_5 = self.get_ratio_following()["not_following_ratio"].dropna()

        fig, axes = plt.subplots(
            3, 1,
            figsize=(40, 24),
            gridspec_kw={"hspace": 0.7},
        )

        fig.suptitle(
            "Exposure Distribution",
            fontsize=self.fontsize_titles,
            y=0.98,
        )

        panels = [
            (
                axes[0], hate_exposure_5, "hate_ratio",
                self.colors["dark_blue"], "Hate Ratio (%)", True, 10, False,
            ),
            (
                axes[1], hate_monthly_5, "monthly_avg_hate",
                self.colors["orange"], "Monthly Average (videos/month)", False, 40, False,
            ),
            (
                axes[2], following_5, "not_following_ratio",
                self.colors["green"], "Not-Following Ratio (%)", True, 40, True,
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
                # A constant distribution cannot be estimated by a KDE.
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

                # Percentage of users per bin.
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

                # KDE density -> percentage per histogram-bin width.
                # This gives the curve and bars comparable y-axis units.
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

                    # Rescale the density currently drawn by seaborn.
                    line = ax.lines[-1]
                    line.set_ydata(line.get_ydata() * bin_width * 100)
                    ax.fill_between(line.get_xdata(), line.get_ydata(),
                                    color=color, alpha=0.1, zorder=2)

                # Include a marker for the mean/median.
                stat_x = np.median(x) if use_median else np.mean(x)
                ax.axvline(
                    stat_x,
                    linestyle="--",
                    color=color,
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"Median = {stat_x:.1f}" if use_median else f"Mean = {stat_x:.1f}",
                )

            ax.set_xlabel(xlabel, fontsize=self.fontsize_labels)
            ax.set_ylabel("% of User IDs per bin", fontsize=self.fontsize_labels)
            ax.tick_params(axis="both", labelsize=self.fontsize_labels)
            ax.xaxis.set_major_locator(MaxNLocator(self.axesticks))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(
                fontsize=self.fontsize_labels * 0.7,
                loc="upper right",
            )

        for ax, values, _, color, xlabel, percent_x, bins, use_median in panels:
            plot_percentage_distribution(
                ax, values, color, xlabel, percent_x=percent_x, bins=bins,
                use_median=use_median
            )

        # Ratios displayed as percentages occupy 0–100 on the x-axis.

        axes[0].set_xlim(0, 10)
        axes[0].set_ylim(0)

        axes[1].set_xlim(0, 30)

        axes[2].set_xlim(100, 80)

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


if __name__=='__main__':
    graph = Exposure_Graph()
    graph.create_exposure()
