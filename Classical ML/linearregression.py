import torch
from utils.optimizers import SGD

class LinearRegression():
    def __init__(self, num_inputs,lr, sigma=0.0):
        self.w = torch.normal(0,sigma, (num_inputs,1), requires_grad=True)
        self.b = torch.zeros(1,requires_grad=True)
        self.lr = lr
    def foward(self,X):
        return torch.matmul(X,self.w) + self.b
    def loss(self, y_hat, y):
        l = ((y_hat - y)**2)/2
        return l.mean()

    def fit_epoch(self, X, y, batch_size):
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            w_batch = self.w[i:i+batch_size]
            y_hat = self.foward(X_batch)
            loss = self.loss(y_hat, y_batch)
            loss.backward()
            opt = SGD([w_batch,self.b],self.lr)
            with torch.no_grad():
                opt.step()
                opt.zero_grad()
        y_hat = self.foward(X)
        loss_total = self.loss(y_hat,y)



