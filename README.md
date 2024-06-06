# Performance package

This package is a practice project for learning how to create a Python package. It is not intended to be a state-of-the-art solution ( all methods we used are based on scikit-learn metric modules). The primary aim is to provide hands-on experience in packaging Python code. This package can ease the assessment of the performance of machine learning models, both classification and regression tasks. For classification tasks, users can assess the performance of models with these metrics:

- accuracy
- precision
- recall
- f1 score
- confusion matrix
- ROC curve and AUC

For regression tasks, users can assess the performance of the models with these metrics:

- root of MSE
- MAE
- R2 score
- scatter plot of y_pred and y_true

This package is not available on PyPI for now. You can clone the directory navigate to the directory and then run 

```
pip install -e .
```
This will install the package on the environment. To utilize it, you just need to import `ClassificationMetrics` (`RegressionMetrics`) class from `performance.classification` module (`performance.regression`) to your workspace and then create an object from this class. The `metric_table` method is responsible for providing the necessary metrics. Take a look at the following example:

```
#classifcation task on iris dataset
from performance.classification import ClassificationMetrics
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#load data
df = sns.load_dataset('iris')

# Encode labels in column 'species'
label_encoder = LabelEncoder()
df['species']= label_encoder.fit_transform(df['species'])

#Split data to train and test
X  = df.iloc[:, :-1]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= .20)

#train the model
lr = LogisticRegression()
lr.fit(X_train, y_train)

#create an instance of the ClassificationMetrics class
cm = ClassificationMetrics()
#determine the data
cm.get_data(X_train, X_test, y_train, y_test)
#model evaluation
cm.metric_table(model = lr, average = 'micro' )

```
