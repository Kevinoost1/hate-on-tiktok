import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import krippendorff
from database import *


"""
*This Script contains the code for the validation procedure and testing procedure.*

The two main metrics used for both testing and validating are:
- ROC curves (with AUC)
- Krippendorffs alpha

plot_roc(): ROC plots (with AUC) are given per class, with seperate lines for each validation results dataframe (to compare).
    - NOTE: the if-statement in this function represents a distinction between binary classification and multi-class classification
      Due to a mistake at the start of this process, I only retrieved the 'prob' column from the early validations' results_df, which only contains the majority class probability.
      In order to keep comparing validation attempts with the earliest runs, without requiring a rerun, I made a (sort of messy) if-statement to fix it.
      However, if you retrieve the 'probs' column from ChoiceLlm the else clause should work for both binary and multi-class classification, making the conditional obsolete.

get_classification_report(): calculates krippendoffs alpha and metrics, such as F1 score and dumps them into a .json matching the specific classification
    - for intolerance and impolitness, RMSE and interval Krippendorffs alpha metrics are calculated instead. 
      In case your classification types are different, but also on a likert scale, you can adjust the names in the list for this function. 

perform_validation(): a helper function that calls all the validation methods for a given classification type for each itteration of the prompt. 
"""


class Validation:
    def __init__(self, validationDfs: list[pd.DataFrame, ...], posLabels: list=['News, Politics & Societal Issues', 'Other'], classificationType: str='topic'):
        self.validationDfs: list[pd.DataFrame, ...] = validationDfs
        self.color_palette: list[str] = ['#001157', '#ec8e3a','#4F8B4C', '#b03aa3', '#2ca7d8', '#a0892c']
        self.posLabels: list[str] = posLabels
        self.classificationType: str = classificationType

    def plot_roc(self, yTrueColumnName: str=f'manual_topic', yScoresColumnName: str='prob') -> None:
        nCols: int = 1 if len(self.posLabels) == 1 else 2
        nRows: int = np.ceil(len(self.posLabels) / 2).astype(int)

        fig, axes = plt.subplots(nRows, nCols)
        axes = np.atleast_1d(axes).flatten()
        for classIdx, label in enumerate(self.posLabels):
            ax = axes[classIdx]
            for dfIdx, df in enumerate(self.validationDfs):

                y_true = (df[yTrueColumnName].apply(lambda x: x.strip()) ==  label).astype(int)

                y_scores_df: pd.DataFrame = df['probs'].str.split(';', expand=True).astype(float)
                y_scores_df.columns = self.posLabels
                y_scores = y_scores_df[label].values

                fpr, tpr, thresholds = roc_curve(y_true, y_scores, drop_intermediate=False)
                auc = roc_auc_score(y_true, y_scores)

                ax.plot(fpr, tpr, label=f'df: {dfIdx}, (AUC: {auc:.3f})')

            ax.plot([0, 1], [0, 1], linestyle='--')
            ax.set_title(f'class: {label}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
        plt.tight_layout()
        plt.show()

    def get_classification_report(self, yTrueColumnName: str=f'manual_topic', yPredColumnName: str='pred') -> dict[int, dict]: 
        reports: dict = {}
        for i, df in enumerate(self.validationDfs):
            df[yTrueColumnName] = df[yTrueColumnName].apply(lambda x: str(x).strip())
            df[yPredColumnName] = df[yPredColumnName].apply(lambda x: str(x).strip())

            if self.classificationType in ['impoliteness', 'intolerance', 'intolerance_test', 'impoliteness_test']:
                rmse = root_mean_squared_error(df[yTrueColumnName], df[yPredColumnName])
                kripp_data = np.array([np.round(df[yTrueColumnName].astype(float), 0), np.round(df[yPredColumnName].astype(float), 0)])
                alpha = krippendorff.alpha(reliability_data=kripp_data, level_of_measurement='ordinal')

                reports[i] = {'rmse': rmse, 'krippendorff_alpha': alpha}

            else:
                class_report = classification_report(df[yTrueColumnName], df[yPredColumnName], target_names=self.posLabels, output_dict=True) 

                le = LabelEncoder()
                le.fit(pd.concat([df[yTrueColumnName], df[yPredColumnName]])) 
                kripp_data = np.array([le.transform(df[yTrueColumnName]), le.transform(df[yPredColumnName])])
                alpha = krippendorff.alpha(reliability_data=kripp_data, level_of_measurement='nominal')

                reports[i] = {'classification_report': class_report, 'krippendorff_alpha': alpha}

        with open(f'classification_report_{self.classificationType}.json', 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=4)

        return reports


def perform_validation(variant_names: list[str, ...], classificationType='topic', ground_truth_name: str='val.csv', manual_label_name: str='manual_topic', 
                       posLabels=['News, Politics & Societal Issues', 'Other']):
    ground_truth: pd.DataFrame = pd.read_csv(f"/home/kevin/Documents/phd/Project-Data-Donation/testset/{ground_truth_name}")
    merged_dfs = []

    for i, v in enumerate(variant_names):
        df = pd.read_csv(f"/home/kevin/Documents/phd/Project-Data-Donation/testset/{v}")
        merged_dfs.append(ground_truth.merge(df, on='video_id', how='inner'))

    val = Validation(merged_dfs, classificationType=classificationType, posLabels=posLabels)

    if classificationType not in ['impoliteness', 'intolerance', 'topic', 'intolerance_test', 'impoliteness_test']:
        val.plot_roc(manual_label_name)
        
    val.get_classification_report(yTrueColumnName=manual_label_name, yPredColumnName='pred')




if __name__ == '__main__':
    perform_validation(variant_names=['top_test_results.csv'], classificationType='topic_test', ground_truth_name='testset_top.csv', manual_label_name='manual_topic')

    perform_validation(variant_names=['imp_test_results.csv'], classificationType='impoliteness_test', ground_truth_name='testset_imp.csv', manual_label_name='manual_impoliteness')









