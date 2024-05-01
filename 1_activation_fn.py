'''1. Write a Python program to plot a few activation functions that are being used in neural networks.
'''


**ACTIVATION** **FUNCTIONS**



import numpy as np
import matplotlib.pyplot as plt

--STEP FUNCTION--



def step(x):
    if x>=0:
        return 1
    else:
        return 0

x = np.linspace(-5,5)
y = [step(i) for i in x]
print('X-values: ',x)
print('Y-values: ',y)
plt.plot(x,y)
plt.xlabel('X value')
plt.ylabel('Step(x)')
plt.title('Step Function')
plt.grid(True)

plt.show()

--SIGMOID FUNCTION--

x = np.linspace(-10,10)
y = (1/(1+np.exp(-x)))


print('X-values: ',x)
print('Y-values: ',y)

plt.plot(x,y)
plt.xlabel('X value')
plt.ylabel('Sigmoid(x)')
plt.title("Sigmoid functon")
plt.grid(True)
plt.show()

--RELU FUNCTION--

x = np.linspace(-10,10)
y = [max(0,i) for i in x]
print('X-values: ',x)
print('Y-values: ',y)
plt.plot(x,y)
plt.xlabel('X value')
plt.ylabel('ReLU(x)')
plt.title("ReLU functon")
plt.grid(True)
plt.show()

 --LEAKY RELU--
 
def leaky_relu(x):
    alpha = 0.1
    return np.where(x>=0,x,alpha*x)


x = np.linspace(-10,10)
y = leaky_relu(x)
#print('X-values: ',x)
#print('Y-values: ',y)
plt.plot(x,y)
plt.xlabel('X value')
plt.ylabel('Leaky ReLU(x)')
plt.title("Leaky ReLU functon")
plt.grid(True)
plt.show()

--TANH FUNCTION--

x = np.linspace(-10,10)
y = 2/(1+np.exp(-2*x))
plt.plot(x,y)
plt.xlabel('X value')
plt.ylabel('Tanh(x)')
plt.title("Tanh functon")
plt.grid(True)
plt.show()

--SOFTMAX FUNCTION--

def softmax(x):
    z = (np.exp(x)/np.exp(np.sum(x)))
    return z



x = np.linspace(-10,10)
y = softmax(x)

plt.plot(x,y)
plt.xlabel('X value')
plt.ylabel('Softmax(x)')
plt.title("Softmax functon")
plt.grid(True)
plt.show()

--BINARY STEP--

    H(x) = 
0  ,   if x < 0
0.5 ,  if x = 0
1 ,    if x > 0


def binarystep(x):
    return np.heaviside(x,0.5)

x = np.linspace(-10,10)
y = binarystep(x)

plt.plot(x,y)
plt.xlabel('X value')
plt.ylabel('binary step(x)')
plt.title("binary step functon")
plt.grid(True)
plt.show()

