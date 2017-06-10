import numpy as np

def signmoid(x):
	return 1/(1 + np.exp(-x))

def signmoid_derivative(x):
	return x*(1-x)

x = np.array([  [0,0,1], [0,1,0], [1,0,1], [1,0,0]] ) #training ip
y = np.array([0,0,1,1]).T  #training op

np.random.seed(1)

w0 = 2*np.random.random((3,1)) - 1

print ("Initial wights : ")
print (w0)

alpha = 1

for i in range(10000):
	#forward
	input_layer = x
	output_layer = signmoid(np.dot(input_layer, w0))

	del_k = (y.T - output_layer) * signmoid_derivative(output_layer)

	del_w = alpha * np.dot(input_layer.T, del_k)

	w0 = del_w + w0

print ("Final weights :")
print (w0)


