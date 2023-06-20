from performance.performance import ModelPerformance
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class RegressionMetrics(ModelPerformance):
    def __init__(self):
        super().__init__()

    def metric_table(self, model, visualize = True):
        self.model = model

        train_performance = []
        test_performance = []

        train_performance.append(np.sqrt(mean_squared_error(self.y_train, self.y_pred_train)))
        test_performance.append(np.sqrt(mean_squared_error(self.y_test, self.y_pred_test)))

        train_performance.append(mean_absolute_error(self.y_train, self.y_pred_train))
        test_performance.append(mean_absolute_error(self.y_test, self.y_pred_test))

        train_performance.append(r2_score(self.y_train, self.y_pred_train))
        test_performance.append(r2_score(self.y_test, self.y_pred_test))

        df_performance = pd.DataFrame([train_performance,test_performance], columns=['RootMSE', 'MAE', 'r2_score'], index= ['trian','test'])
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