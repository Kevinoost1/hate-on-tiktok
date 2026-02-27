import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import krippendorf


class Validation:
    def __init__(self, *validationDf: pd.DataFrame, posLabels: list=['News, Politics & Societal Issues']):
        self.validationDfs: tuple[pd.DataFrame, ...] = validationDf
        self.color_palette: list[str] = ['#001157', '#ec8e3a','#4F8B4C', '#b03aa3', '#2ca7d8', '#a0892c']
        self.posLabels: list[str] = posLabels

    def plot_roc(self, yTrueColumnName: str='y_true', yScoresColumnName: str='y_pred') -> None:
        nCols = 1 if len(self.posLabels) == 1 else 2
        nRows = len(self.posLabels) // 2

        fig, axes = plt.subplots(nRows, nCols)
        axes = np.atleast_1d(axes).flatten()
        for classIdx, label in enumerate(self.posLabels):
            ax = axes[classIdx]
            for dfIdx, df in enumerate(self.validationDfs):
                y_true = (df[yTrueColumnName] ==  label).astype(int)
                y_scores = df[yScoresColumnName]

                fpr, tpr, thresholds = roc_curve(y_true, y_scores)
                auc = roc_auc_score(y_true, y_scores)

                ax.plot(fpr, tpr, label=f'df: {dfIdx}, (AUC: {auc:.3f})')

            ax.plot([0, 1], [0, 1], linestyle='--')
            ax.set_title(f'class: {label}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend()
        plt.tight_layout()
        plt.show()

    def get_classification_report(self, yTrueColumnName: str='y_pred', yPredColumnName: str='y_true') -> dict[int, dict]
        reports: dict = {}
        for i, df in enumerate(self.validationDfs):
            class_report = classification_report(df[yTrueColumnName], df[yPredColumnName], 
                                              target_names=self.posLabels, output_dict=True)
            
            kripp_data = np.array([df[yTrueColumnName].values, df[yPredColumnName].values])
            alpha = krippendorff.alpha(reliability_data=kripp_data, level_of_measurement='nominal')

            reports[i] = {'classification_report': class_report, 'krippendorff_alpha': alpha}


        with open('classification_report.json', 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=4)

        return reports



