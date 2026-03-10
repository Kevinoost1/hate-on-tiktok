import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import json
import krippendorff
from database import *



class Validation:
    def __init__(self, *validationDf: pd.DataFrame, posLabels: list=['News, Politics & Societal Issues', 'Other'], classificationType: str='topic'):
        self.validationDfs: tuple[pd.DataFrame, ...] = validationDf
        self.color_palette: list[str] = ['#001157', '#ec8e3a','#4F8B4C', '#b03aa3', '#2ca7d8', '#a0892c']
        self.posLabels: list[str] = posLabels
        self.classificationType: str = classificationType

    def plot_roc(self, yTrueColumnName: str=f'manual_topic', yScoresColumnName: str='prob') -> None:
        nCols = 1 if len(self.posLabels) == 1 else 2
        nRows = np.ceil(len(self.posLabels) / 2).astype(int)
        print(nRows)

        fig, axes = plt.subplots(nRows, nCols)
        axes = np.atleast_1d(axes).flatten()
        for classIdx, label in enumerate(self.posLabels):
            ax = axes[classIdx]
            for dfIdx, df in enumerate(self.validationDfs):
                y_true = (df[yTrueColumnName].apply(lambda x: x.strip()) ==  label).astype(int)
                if self.classificationType == 'topic':
                    y_scores = np.where(df['pred'] == label, df['prob'].astype(float), 1 - df['prob'].astype(float))
                else:
                    y_scores_df: pd.DataFrame = df['probs'].str.split(';', expand=True).astype(float)
                    y_scores_df.columns = self.posLabels
                    y_scores = y_scores_df[label].values
                    print(y_scores)
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

    def get_classification_report(self, yTrueColumnName: str=f'manual_topic', yPredColumnName: str='pred') -> dict[int, dict]: 
        reports: dict = {}
        for i, df in enumerate(self.validationDfs):
            df[yTrueColumnName] = df[yTrueColumnName].apply(lambda x: x.strip())
            df[yPredColumnName] = df[yPredColumnName].apply(lambda x: x.strip())

            print(df[yTrueColumnName].unique())
            print(df[yPredColumnName].unique())
            class_report = classification_report(df[yTrueColumnName], df[yPredColumnName], 
                                              target_names=self.posLabels, output_dict=True) 
            le = LabelEncoder()
            le.fit(pd.concat([df[yTrueColumnName], df[yPredColumnName]]))
            print(df[yTrueColumnName], df[yPredColumnName])
            kripp_data = np.array([le.transform(df[yTrueColumnName]), le.transform(df[yPredColumnName])])
            alpha = krippendorff.alpha(reliability_data=kripp_data, level_of_measurement='nominal')

            reports[i] = {'classification_report': class_report, 'krippendorff_alpha': alpha}


        with open(f'classification_report_{self.classificationType}.json', 'w', encoding='utf-8') as f:
            json.dump(reports, f, indent=4)

        return reports


def topic():
    df_val2 = pd.read_csv("/home/kevin/Documents/phd/Project-Data-Donation/testset/val_variant2.csv")
    df_val3 = pd.read_csv("/home/kevin/Documents/phd/Project-Data-Donation/testset/val_variant3.csv")
    df_val4 = pd.read_csv("/home/kevin/Documents/phd/Project-Data-Donation/testset/val_variant4.csv")
    df_val5 = pd.read_csv("/home/kevin/Documents/phd/Project-Data-Donation/testset/val_variant5.csv")
    df_val6 = pd.read_csv("/home/kevin/Documents/phd/Project-Data-Donation/testset/val_variant6.csv")
    df_val7 = pd.read_csv("/home/kevin/Documents/phd/Project-Data-Donation/testset/val_variant7.csv")
    df_val8 = pd.read_csv("/home/kevin/Documents/phd/Project-Data-Donation/testset/val_variant8.csv")
    df_val9 = pd.read_csv("/home/kevin/Documents/phd/Project-Data-Donation/testset/val_variant9.csv")

    df_orig = pd.read_csv('/home/kevin/Documents/phd/Project-Data-Donation/testset/val.csv')

    df2 = df_orig.merge(df_val2, on='video_id', how='left')    
    df3 = df_orig.merge(df_val3, on='video_id', how='left')    
    df4 = df_orig.merge(df_val4, on='video_id', how='left')    
    df5 = df_orig.merge(df_val5, on='video_id', how='left')    
    df6 = df_orig.merge(df_val6, on='video_id', how='left')    
    df7 = df_orig.merge(df_val7, on='video_id', how='left')    
    df8 = df_orig.merge(df_val8, on='video_id', how='left')    
    df9 = df_orig.merge(df_val9, on='video_id', how='left')    
    
    val = Validation(df2, df3, df4, df5, df6, df7, df8, df9)
    val.plot_roc('manual_topic', 'prob')
    val.get_classification_report('manual_topic', 'pred')

def sent():
    df_val1 = pd.read_csv("/home/kevin/Documents/phd/Project-Data-Donation/testset/val_sent_variant1.csv")
    print(df_val1.columns)

    df_orig = pd.read_csv('/home/kevin/Documents/phd/Project-Data-Donation/testset/val.csv')
    df1 = df_orig.merge(df_val1, on='video_id', how='left').iloc[:150, :]  

    val = Validation(df1, posLabels=['Negative', 'Neutral', 'Positive'], classificationType='sent')
    val.plot_roc('manual_sent', 'prob')
    val.get_classification_report('manual_sent', 'pred')

if __name__ == '__main__':
    sent()
    topic()

