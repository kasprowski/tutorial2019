'''
Example of linear regression using gradient descent
compared with the analytical method

@author: pawel@kasprowski.pl
based on:
https://towardsdatascience.com/linear-regression-using-gradient-descent-97a6c8700931
'''

import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
X = data[:, 0]
Y = data[:, 1]
plt.scatter(X, Y, label="Points")

# initial values
a = 0
b = 1

# another example - b value is closer to ideal
#a = 2
#b = 9

# another example - b value is far from ideal
#a = 5
#b = -9



L = 0.0001  # The learning Rate (for 0.0004 - jumps over the solution)
epochs = 10  # The number of iterations to perform gradient descent

n = float(len(X)) # Number of elements in X

a_values = []
b_values = []


# Performing Gradient Descent 
for i in range(epochs): 
    print()
    print("[{}]".format(i))
    a_values.append(a)
    b_values.append(b)

    print("a={:.3f} b={:.3f}".format(a,b))

    Y_pred = a*X + b  # The current predicted value of Y
    D_a = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt a
    D_b = (-2/n) * sum(Y - Y_pred)  # Derivative wrt b
    print(" D_a={:.3f} D_b={:.3f}".format(D_a,D_b))
    na = a - L * D_a  # Update a
    nb = b - L * D_b  # Update b
    print ("New a: {:.3f} - L * {:.3f} = {:.3f}".format(a, D_a, na))
    print ("New b: {:.3f} - L * {:.3f} = {:.3f}".format(b, D_b, nb))
    a = na
    b = nb
    # Make new prediction
    Y_pred = a*X + b
    color = (i/epochs,0,0)
    plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color=color, label=str(i))  # regression line

print("=====================================")
E1 = ((Y-(a*X+b))**2).mean()
print("Found coeficients: a={:.3f} b={:.3f}, error={:.5f}".format(a,b,E1))

# Calculate coefficients (ca, cb) analytically
avg_X = np.average(X)
avg_Y = np.average(Y)
numerator = np.sum((X-avg_X)*(Y-avg_Y))
denominator = np.sum((X-avg_X)*(X-avg_X)) 
ca =numerator/denominator
cb = avg_Y - ca*avg_X
Y_pred = ca*X + cb  
E2 = ((Y-Y_pred)**2).mean()
print("Calculated coeficients: a={:.3f} b={:.3f}, error={:.5f}".format(ca,cb,E2))
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color=(0,0,1), label="calc")  # calculated line


plt.legend(loc='lower left')    
plt.show()

plt.plot(range(len(a_values)),a_values,label="A values")
#plt.plot(range(len(b_values)),b_values,label="B values")
plt.legend(loc='lower left')    
plt.show()