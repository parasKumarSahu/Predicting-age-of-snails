import numpy as np
import random as rn


def standardize(X, mean, std):
	if mean == None and std == None:
		mean = list(np.mean(X, axis=0))
		std = list(np.std(X, axis=0))
	for i in range(len(X)): 
		for j in range(1, len(X[0])):
			X[i][j] = float(X[i][j] - mean[j])/float(std[j]+0.0000000001)
	return mean, std		

def mylinridgereg(X, y, lamda):
	Xt = np.transpose(X)	
	first = np.linalg.pinv(np.matmul(Xt,X)+(lamda*np.identity( len(X[0]) )))
	return np.matmul( first, np.matmul(Xt,y) ) 

def split_train_test(X, y, frac):
	X_train = []
	y_train = []
	X_test = []
	y_test = []
	for i in range(len(X)):
		if rn.random() < frac:
			X_train.append(X[i])
			y_train.append(y[i])
		else:
			X_test.append(X[i])
			y_test.append(y[i])	
	return X_train, X_test, y_train, y_test		

def mylinridgeregeval(X_test, w):
	return np.matmul(X_test, np.transpose(w))

def meansquarederr(ans, y_test):
	return ((ans - y_test) ** 2).mean()


#Main Function

f = open("linregdata", "r")
#output
y = []
#Training Data
X = []
#lambda
lamda = 1
#Fraction of training/validation set
frac = 0.2

for i in f:
	tmp = i[:-1].split(",")
	y.append(int(tmp[-1]))
	if tmp[0] == 'F':
		X.append([1,1,0,0]+[float(i) for i in tmp[1:-1]])
	if tmp[0] == 'I':
		X.append([1,0,1,0]+[float(i) for i in tmp[1:-1]])
	if tmp[0] == 'M':
		X.append([1,0,0,1]+[float(i) for i in tmp[1:-1]])

X_train, X_test, y_train, y_test = split_train_test(X, y, frac)
mean, std = standardize(X_train, None, None)
standardize(X_test, mean, std)

w = mylinridgereg(X_train, y_train, lamda)
ans =  mylinridgeregeval(X_test, w)
#print(y)
#print(X[0])
print(meansquarederr(ans, y_test))