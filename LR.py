import numpy as np
from sklearn.linear_model import LinearRegression


class LR:
    def __init__(self,x,t):
        self.x = x
        self.t = t
        assert self.x.shape[0] == self.t.shape[0], "配列の形状が一致しません"
        self.model = self.fitting()
        self.score = self.model.score(self.x,self.t)
        print(f"score : {self.score}")
        self.coef = self.model.coef_
        print(f"coef : {self.coef}")
        self.intercept = self.model.intercept_



    def fitting(self):
        model = LinearRegression()
        model.fit(self.x,self.t)
        return model
    
    def predict(self,x):
        pred = self.model.predict(x)
        return pred

if __name__ == "__main__":
    x = np.array([[1,2,3],[4,5,6]])
    t = np.array([[1],[2]])
    print(x.shape)
    print(t.shape)
    model = LR(x,t)
