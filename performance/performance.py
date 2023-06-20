from abc import ABC, abstractmethod


class ModelPerformance(ABC):
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def get_data( self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test =  y_test

    def get_model(self, model):
        self.model = model

    @property
    def y_pred_train(self):
        return self.model.predict(self.X_train)
    
    @property
    def y_pred_test(self):
        return self.model.predict(self.X_test)

    @abstractmethod
    def metric_table(self):
        pass
    
