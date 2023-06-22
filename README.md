# Performance package

The aim of this package is to ease the assessment of the performance of machine learning models, both classification and regression tasks. All methods we used are based on scikit-learn metric modules. For classification tasks, users can assess the performance of their models with these metrics:

 -accuracy
 -precision
 -recall
 -f1_score
 -confusion matrix
 -ROC curve and AUC
 
To use this package you just need to import performance module to your workspace and then create an object from the `Performance` class passinng the name of machine learning model and machine learning task type. The is an example:

```
from ml-perfomance import Performance
model_per = Performance(task_type = 'r', model = model)

```
