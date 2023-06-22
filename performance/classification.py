from typing import Union
from performance.performance import ModelPerformance
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, auc, f1_score, recall_score, precision_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ClassificationMetrics(ModelPerformance):
    def __init__(self):
        super().__init__()
    
    def metric_table(self, model, confusion_matrix = True, average= 'binary', metric: Union[str, list] = ['percision', 'recall', 'f1']):

        """
        This method creates a DataFrame of the selected metrics

        model (ML estimator): Fitted ML model

        Confusion_matrix (binary): {True, False}. If True, it returns the confusion matrix of train and test data, default= True

        average (str): {'micro', 'macro', 'binary'} or None, default= 'binary'. For multi-class classification it should be one of 'mirco', 'macro' or None. 
        To see these parameters descriptions, visit sklearn.metrics.percision_score() documnetation.

        metric (Union[str, list]): A list of ['percision', 'recall', 'f1'] or one of them. It always includes accuracy by default.
        """

        self.model = model
        train_performance = []
        test_performance = []

        train_performance.append(accuracy_score(self.y_train, self.y_pred_train))
        test_performance.append(accuracy_score(self.y_test, self.y_pred_test))
        columns = ['accuracy']

        metric_dic = {'percision': precision_score, 'recall': recall_score, 'f1': f1_score}

        if isinstance(metric, str):
            metric =[metric]
        
        if average !=None:
            for m in metric:
                train_performance.append(metric_dic[m](self.y_train, self.y_pred_train,  average = average))
                test_performance.append(metric_dic[m](self.y_train, self.y_pred_train,  average = average))
                columns.append(m)

        else:
            labels = self.y_train.unique()
            labels.sort()

            for m in metric:    
                train_performance.extend(metric_dic[m](self.y_train, self.y_pred_train, average = None))
                test_performance.extend(metric_dic[m](self.y_train, self.y_pred_train, average = None))

                for label in labels:
                    columns.append( m +"_"+ str(label))

        performance_df = pd.DataFrame([train_performance,test_performance], columns=columns, index= ['trian','test'])
        display(performance_df)

        if confusion_matrix ==True:
            #confusion matrix:
            fig, ax = plt.subplots(1,2,figsize = (12,5))

            cm_train_display = ConfusionMatrixDisplay.from_estimator(self.model, self.X_train, self.y_train, ax= ax[0])
            ax[0].set_title("Confusion matrix of Train data")

            cm_test_display = ConfusionMatrixDisplay.from_estimator(self.model, self.X_test, self.y_test, ax = ax[1] )
            ax[1].set_title("Confusion matrix of Test data")

            plt.show()
        
    
    def roc_curve(self, model):
        self.model = model

        precision_train, recall_train, thresholds_train = precision_recall_curve(self.y_train, self.y_pred_train)
        precision_test, recall_test, thresholds_test = precision_recall_curve(self.y_test, self.y_pred_test)
        auc_pr_train = auc(recall_train, precision_train)
        auc_pr_test = auc(recall_test, precision_test)

        fig , ax = plt.subplots(1,2 , figsize= (12,5))
        ax[0].plot(recall_train, precision_train, label='AUC: {:.3f}'.format(auc_pr_train))
        ax[0].set_xlabel('Recall')
        ax[0].set_ylabel('1 - Precision')
        ax[0].set_title('ROC curve- Train data')
        ax[0].legend()

        ax[1].plot(recall_test, precision_test, label='AUC: {:.3f}'.format(auc_pr_test))
        ax[1].set_xlabel('Recall')
        ax[1].set_ylabel('1 -Precision')
        ax[1].set_title('ROC curve- Test data')
        ax[1].legend()

        plt.show()
