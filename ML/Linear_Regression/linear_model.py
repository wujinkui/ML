import numpy as np

class LinearRegression():
    """This is the Linear model we will use two strategies to compelete this task
    Given a matrix X and a target vector y, maximum-likelihood estimate"""

    def __init__(self):
        self.intercept = None
        self.theta = None

    def fit(self,X,y):
        """
        X, and y should be np.ndarray
        """
        X = np.c_[np.ones(X.shape[0]),X]
        all_parameters = np.linalg.inv(X.T@X)@X.T@y
        self.intercept = all_parameters[0]
        self.theta = all_parameters[1:]


    def predict(self,X):
        return X@self.theta + self.intercept*np.ones([X.shape[0],1])






