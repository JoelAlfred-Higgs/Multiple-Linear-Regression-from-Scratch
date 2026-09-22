import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
class HousePricing:
    def __init__(self):
        data = pd.read_csv(r"C:\Users\joela\OneDrive\Documents\Regression\House Price Prediction Dataset.csv")
        self.x_raw = data['Area'].to_numpy()
        self.y = data['Price'].to_numpy()

        self.x_mean = np.mean(self.x_raw)
        self.x_std = np.std(self.x_raw)
        self.x = (self.x_raw - self.x_mean) / self.x_std

    def compute_cost(self,w,b):
        m = len(self.x)
        cost = 0
        f_x = w * self.x + b
        cost = (1 / (2 * m)) * np.sum((f_x - self.y) ** 2)
        return cost
    
    def compute_gradient(self,w,b):
        m = len(self.x)
        dj_dw = 0
        dj_db = 0
        f_x = w*self.x + b
        dj_dw = (1 / m) * np.sum((f_x - self.y) * self.x)
        dj_db = (1 / m) * np.sum(f_x - self.y)
        return dj_dw , dj_db
    
    def gradient_descent(self,w,b,alpha,iteration):
        m = len(self.x)
        w_history = []
        j_history = []
        
        for i in range(iteration):
            dj_dw,dj_db = self.compute_gradient(w,b)
            w = w - alpha*dj_dw
            b = b - alpha*dj_db
            if i % 10 == 0:
                j_history.append(self.compute_cost(w,b))
                w_history.append(w)
        return w,b,j_history,w_history

    def predict(self,raw_input,w,b):
        processed_input  = (raw_input - self.x_mean)/self.x_std
        return (w*processed_input + b)
        
houseprice = HousePricing()
w_init = 0
b_init = 0
alpha = 0.05
iterations = 1000
area  = int(input("Enter area of the House: "))

w,b,j_hist,w_hist = houseprice.gradient_descent(w_init,b_init,alpha,iterations)
resultant_price = houseprice.predict(area,w,b)
print(f"Resultant price: {resultant_price}")
plt.plot(j_hist)
plt.xlabel("Iterations (x10)")
plt.ylabel("Cost")
plt.title("Convergence of Gradient Descent")
plt.plot(x)
plt.show()
