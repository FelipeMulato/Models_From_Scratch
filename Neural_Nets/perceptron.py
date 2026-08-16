import numpy as np

class Perceptron:
    def __init__(self,num_inputs, lr=0.0005):
        self.lr = lr
        self.W = np.random.uniform(-2,2,num_inputs)
        self.B = np.random.uniform(-2,2)

    def activate_function(self, z):
        if z>=0:
            return 1
        else:
            return 0
    def foward(self, x):
        return self.activate_function(np.dot(self.W,x)+ self.B)
    
    def update(self,erro,x):
        self.W = self.W - self.lr*erro*x
        self.B = self.B - self.lr*erro
    def train(self,X,Y,epochs):
        size = len(X)
        print(size)
        print(f"Pesos iniciais {self.W} e bias inicial {self.B}")
        for i in range(0,epochs):
            for j in range(0,size):
                x = X[j]
                y = Y[j]
                y_hat = self.foward(x)
                self.update(y_hat-y,x)
                erro_total = 0
            for k in range(0,size):
                x = X[k]
                y = Y[k]
                y_hat = self.foward(x)
                erro_total+=abs(y-y_hat)
                acuracia = 1- erro_total/size
            print(f"A taxa de acerto depois da Epoca {i+1} foi de {acuracia:.2f}")                

        
        
        print(f"Pesos finais {self.W} e bias final {self.B}")



                



    
