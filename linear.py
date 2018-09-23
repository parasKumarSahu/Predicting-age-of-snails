import numpy as np
import random as rn
import matplotlib.pyplot as plt

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
lamda = [i for i in range(10)]
#Fraction of training/validation set
fraction = [0.1+(i*9)/1000 for i in range(100)]

for i in f:
	tmp = i[:-1].split(",")
	y.append(int(tmp[-1]))
	if tmp[0] == 'F':
		X.append([1,1,0,0]+[float(i) for i in tmp[1:-1]])
	if tmp[0] == 'I':
		X.append([1,0,1,0]+[float(i) for i in tmp[1:-1]])
	if tmp[0] == 'M':
		X.append([1,0,0,1]+[float(i) for i in tmp[1:-1]])

frac_no = 0
min_error = []
min_error_lamda = []

min_error_predictions_train = []
min_error_predictions_test = []
min_error_actual_train_values = []
min_error_actual_test_values = []

for frac in fraction:
	X_train, X_test, y_train, y_test = split_train_test(X, y, frac)
	mean, std = standardize(X_train, None, None)
	standardize(X_test, mean, std)

	train_error = []
	test_error = []
	for l in lamda:
		w = mylinridgereg(X_train, y_train, l)
		ans =  mylinridgeregeval(X_train, w)
		train_error.append(meansquarederr(ans, y_train))	
		ans =  mylinridgeregeval(X_test, w)
		test_error.append(meansquarederr(ans, y_test))	

	plt.xlabel('Lambda Values') 
	plt.ylabel('Mean Square Error') 
	plt.title('Number of training examples '+str(len(X_train))+' total examples '+str(len(X))) 

	plt.plot(lamda, train_error, color='green', linestyle='dashed', linewidth = 1, 
	         marker='o', markerfacecolor='blue', markersize=5) 
	plt.plot(lamda, test_error, color='red', linestyle='dashed', linewidth = 1, 
	         marker='o', markerfacecolor='blue', markersize=5) 
	plt.legend(["Training", "Test"])
	plt.savefig("graphs/p"+str(frac_no)+".png")
	plt.clf()
	frac_no += 1
	min_error.append(min(test_error))
	min_error_lamda.append(test_error.index(min(test_error)))
	w = mylinridgereg(X_train, y_train, test_error.index(min(test_error)))
	ans =  mylinridgeregeval(X_train, w)
	min_error_predictions_train.append(ans)
	ans =  mylinridgeregeval(X_test, w)
	min_error_predictions_test.append(ans)
	min_error_actual_train_values.append(y_train)
	min_error_actual_test_values.append(y_test)


plt.xlabel('Fraction Values') 
plt.ylabel('Min Mean Square Error') 
plt.title('Min Mean Square Error Vs Data Fraction') 

plt.plot(fraction, min_error, color='green', linestyle='dashed', linewidth = 1, 
         marker='o', markerfacecolor='blue', markersize=5) 
plt.savefig("mseVsfraction.png")
plt.clf()

plt.xlabel('Fraction Values') 
plt.ylabel('Lambda for min Mean Square Error') 
plt.title('Lambda for min Mean Square Error Vs Data Fraction') 

plt.plot(fraction, min_error_lamda, color='green', linestyle='dashed', linewidth = 1, 
         marker='o', markerfacecolor='blue', markersize=5) 
plt.savefig("lambdaVSfraction.png")
plt.clf()

best_lamda = min_error_lamda[min_error.index(min(min_error))]
best_fraction = 0.1+(min_error.index(min(min_error))*9)/1000
plt.xlabel('Predicted Age of snails') 
plt.ylabel('Actual Age of snails') 
plt.title('Min Training Error fraction: '+str(best_fraction)+" lambda: "+str(best_lamda)+" Error: "+str(min(min_error))) 
plt.scatter(min_error_actual_train_values[best_lamda], min_error_predictions_train[best_lamda]) 
plt.plot([i for i in range(40)], [i for i in range(40)], color='green', linestyle='dashed', linewidth = 1) 
plt.savefig("predictedVSgivenTrain.png")
plt.clf()

plt.xlabel('Predicted Age of snails') 
plt.ylabel('Actual Age of snails') 
plt.title('Min Test Error fraction: '+str(best_fraction)+" lambda: "+str(best_lamda)+" Error: "+str(min(min_error))) 
plt.scatter(min_error_actual_test_values[best_lamda], min_error_predictions_test[best_lamda]) 
plt.plot([i for i in range(40)], [i for i in range(40)], color='green', linestyle='dashed', linewidth = 1) 
plt.savefig("predictedVSgivenTest.png")
plt.clf()
