# Performance package

The aim of this package is to ease the assessment of the performance of machine learning models, both classification and regression task. 
The current version (0.0.1) can be applied just on binary classifiction tasks. To use this package you just need to import Performance module to your workspace and then create an object from the `Performance` class passinng the name of machine learning model and machine learning task type. The is an example:

```
from ml-perfomance import Performance
model_per = Performance(task_type = 'r', model = model)

```
