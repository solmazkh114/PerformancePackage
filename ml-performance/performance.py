from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix, precision_recall_curve, auc, f1_score, recall_score, precision_score, accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ModelPerformance:
    def __init__(self, task_type, model):
        self.task_type = task_type
        self.model = model
        self.X_train = globals().get('X_train')
        self.y_train = globals().get('y_train')
        self.X_test = globals().get('X_test')
        self.y_test = globals().get('y_test')


    def set_data(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    @property
    def y_pred_train(self):
        return self.model.predict(self.X_train)
    
    @property
    def y_pred_test(self):
        return self.model.predict(self.X_test)

    # For now, just support binary classification
    def performance_dataframe(self, visualize = True):
        if self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
            raise ValueError("Specify the train and test data using set_data method")
        
        
        if self.task_type == 'r':
            train_performance = []
            test_performance = []

            train_performance.append(np.sqrt(mean_squared_error(self.y_train, self.y_pred_train)))
            test_performance.append(np.sqrt(mean_squared_error(self.y_test, self.y_pred_test)))

            train_performance.append(mean_absolute_error(self.y_train, self.y_pred_train))
            test_performance.append(mean_absolute_error(self.y_test, self.y_pred_test))

            train_performance.append(r2_score(self.y_train, self.y_pred_train))
            test_performance.append(r2_score(self.y_test, self.y_pred_test))

            df_performance = pd.DataFrame([train_performance,test_performance], columns=['Root_mean_squared_error', 'mean_absolute_error', 'r2_score'], index= ['trian','test'])
            display(df_performance)
            
            if visualize== True:
                fig ,ax = plt.subplots(1,2, figsize = (10,4))
                sns.scatterplot(x = self.y_train, y = self.y_pred_train, ax = ax[0])
                ax[0].set_xlabel("y_real")
                ax[0].set_ylabel("y_prediction")
                ax[0].set_title("Train")
                ax[0].plot([self.y_train.min(), self.y_train.max()], [self.y_train.min(), self.y_train.max()], 'r--')

                sns.scatterplot(x = self.y_test, y = self.y_pred_test, ax = ax[1])
                ax[1].set_xlabel("y_real")
                ax[1].set_ylabel("y_prediction")
                ax[1].set_title("Test")
                ax[1].plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
                plt.show()

        if self.task_type=='c':
            train_performance = []
            test_performance = []
            metrics_list = [accuracy_score,precision_score, recall_score, f1_score ]
            for m in metrics_list:
                train_performance.append(m(self.y_train, self.y_pred_train))
                test_performance.append(m(self.y_test, self.y_pred_test))

            
            performance_df = pd.DataFrame([train_performance,test_performance], columns=['accuracy', 'precision', 'recall', 'f1'], index= ['trian','test'])
            display(performance_df)

            if visualize ==True:
                #confusion matrix:
                fig, ax = plt.subplots(1,2,figsize = (12,5))

                cm_train_display = ConfusionMatrixDisplay.from_estimator(self.model, self.X_train, self.y_train, ax= ax[0])
                ax[0].set_title("Confusion matrix of Train data")

                cm_test_display = ConfusionMatrixDisplay.from_estimator(self.model, self.X_test, self.y_test, ax = ax[1] )#display_labels = [False, True])
                ax[1].set_title("Confusion matrix of Test data")

                cm_train_display.confusion_matrix
                cm_test_display.confusion_matrix
                plt.show()
        
    
    #def auc_curve(self):
    #    if self.task_type == 'r':
    #        raise TypeError("For AUC curve, the task type has to be classification")
        
