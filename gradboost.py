import numpy as np
from tqdm import tqdm_notebook
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from matplotlib import pyplot as plt

class GradientBoosting(BaseEstimator):
    
    def sigma(self,z):
        z=z.reshape([z.shape[0],1])
        z[z>100]=100
        z[z<-100]=-100
        return 1./(1+np.exp(-z))
    
    
    def log_loss_grad(self,y,p):
        y=y.reshape([y.shape[0],1])
        p=p.reshape([p.shape[0],1])
        return (p-y) / p / (1-p)
    
    
    def mse_grad(self,y,p):
        return 2*(p-y.reshape([y.shape[0],1]))/y.shape[0]
    
    def rmsle(self,y,p):
        y=y.reshape([y.shape[0],1])
        p=p.reshape([p.shape[0],1])
        return np.mean(np.log((p+1)/(y+1))**2)**0.5
    
    def __init__(self,n_estimators=10,learning_rate=0.01,
                 max_depth=3,random_state=17,loss='mse', debug=False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.loss = loss
        self.debug = debug
        self.initialization = lambda y:np.mean(y)*np.ones([y.shape[0],1])
        
        if loss=='log_loss':
            self.objective = log_loss
            self.objective_grad =self.log_loss_grad
        elif loss == 'mse':
            self.objective = mean_squared_error
            self.objective_grad = self.mse_grad
        elif loss == 'rmsle':
            self.objective = self.rmsle
            self.objective_grad = self.rmsle_grad
        elif loss == 'rmspe':
            self.objective = rmspe
            self.objective_grad = self.rmspe_grad
        
        self.trees_ = []
        self.loss_by_iter =[]
        
        if self.debug:
            self.residuals =[]
            self.temp_pred =[]
            
            
    def fit(self, X, y):
        self.X = X
        self.y = y
        b=self.initialization(y)
        prediction = b.copy()
        
        for t in range(self.n_estimators):
            resid = - self.objective_grad(y,prediction)
            if self.debug:
                self.residuals.append(resid)
                
            tree=DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X,resid)
            b=tree.predict(X).reshape([X.shape[0],1])
            if self.debug:
                self.temp_pred.append(b)
            self.trees_.append(tree)
            prediction +=self.learning_rate*b
            self.loss_by_iter.append(self.objective(y,prediction))
            
        self.train_pred = prediction
    
        if self.objective == log_loss:
            self_train_pred = self.sigma(self.train_pred)
        
        return self
    
    def predict_proba(self,X):
        pred = np.ones([X.shape[0], 1])*np.mean(self.y)
        for t in range(self.n_estimators):
            pred+=self.learning_rate*self.trees_[t].predict(X).reshape([X.shape[0],1])
        return pred
            
    def predict(self,X):
        
        pred_probs = self.predict_proba(X)
        return pred_probs    

X_regr_toy= np.arange(7).reshape(-1,1)
Y_regr_toy=((X_regr_toy-3)**2).astype('float64')
boost_regr_mse=GradientBoosting(n_estimators=200, loss='mse', max_depth=3,learning_rate=0.1, debug=True)
boost_regr_mse.fit(X_regr_toy,Y_regr_toy)
predictions = boost_regr_mse.predict(X_regr_toy).reshape(len(X_regr_toy))
answers = Y_regr_toy.reshape(len(Y_regr_toy))
allerrors = np.abs(predictions - answers)
error = '{percent:.2%}'.format(percent=np.sum(allerrors)/np.sum(answers))
print(error)