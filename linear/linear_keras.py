'''
Deep Learning in the Eye Tracking World tutorial source file
https://www.github.com/kasprowski/tutorial2019

Example of linear regression with gradient descent using Keras library.
The "network" consists of one perceptron with one input, bias and one output

@author: pawel@kasprowski.pl
'''

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

#load data
data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
X = data[:, 0]
Y = data[:, 1]
plt.scatter(X, Y, label="Points")

X = np.array(X)
Y = np.array(Y)

# build model
model = Sequential()
dense = Dense(1,input_shape=(1,))
model.add(dense)
model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mae","mse"])

# fit model
model.fit(X,Y,epochs=500)

# check errors
Y_pred = model.predict(X)[:,0]
E_pred = ((Y-Y_pred)**2).mean()
# read weights from the network 
pa, pb = model.get_weights()
pa = pa[0,0]
pb = pb[0]
print("Predicted coeficients: a={:.3f} b={:.3f}, error={:.5f}".format(pa,pb,E_pred))
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color=(0,0,1), label="predicted")  # predicted line

# Calculation of coefficients
avg_X = np.average(X)
avg_Y = np.average(Y)
ca = np.sum((X-avg_X)*(Y-avg_Y)) / np.sum((X-avg_X)*(X-avg_X))
cb = avg_Y - ca*avg_X
Y_calc = ca*X+cb
E_calc = ((Y-Y_calc)**2).mean()
print("Calculated coeficients: a={:.3f} b={:.3f}, error={:.5f}".format(ca,cb,E_calc))
plt.plot([min(X), max(X)], [min(Y_calc), max(Y_calc)], color=(1,0,0), label="calculated")  # calculated line

plt.legend(loc='lower left')    
plt.show()
