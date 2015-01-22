# Emma Peterson
# edp2117
# AI Homework 3 
# Problem 2: Classification with Support Vector Machines


import sklearn as sk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import svm
from matplotlib import cm
from sklearn.cross_validation import *
from sklearn.metrics import accuracy_score

# open the chessboard dataset
f1 = open('chessboard.csv', 'r')
text = f1.read()

# read the data into a two dimensional array (A, B, label)
data = text.split("\r")[:]
for i in xrange(1, len(data)):
	data[i] = data[i].split(",")[:]
for i in xrange(1, len(data)):
	data[i] = [float(data[i][0]),float(data[i][1]), int(data[i][2])]

data.pop(0)

# plot the data
x0 = [] # A values of points with classifier 0
y0 = [] # B values of points with classifier 0
x1 = [] # A values of points with classifier 1
y1 = [] # B values of points with classifier 1
f1 = [] # all values for feature A
f2 = [] # all values for feature B
for i in xrange(len(data)):
	if 	data[i][2] == 0:
		x0.append(data[i][0])
		y0.append(data[i][1])
x0 = np.array(x0[:])
y0 = np.array(y0[:])
plt.plot(x0,y0, 'bs') # plot class 0 data points as blue circles
for i in xrange(len(data)):
	if 	data[i][2] == 1:
		x1.append(data[i][0])
		y1.append(data[i][1])
x1 = np.array(x1[:])
y1 = np.array(y1[:])
plt.plot(x1,y1, 'g^') # plot class 1 data points as green triangles
plt.xlabel('A')
plt.ylabel('B')
plt.title('Chessboard Plot')
plt.axis([-0.1,4.1,-0.1,4.1]) # adjusted axes

plt.savefig("4.pdf")
plt.show()


# split data into training (60%) and testing (40%)
train, test = train_test_split(data, test_size = 0.40)


# builds support vector machines with different kernels 
def buildClassifier(X, y, x0, y0, x1, y1, f1, f2, train_data, test_data):


	titles = ['SVC with linear kernel', 'SVC with RBF kernel', 'SVC with polynomial kernel']
	f1 = np.array(f1[:]) # a numpy array of all of the training A features
	f2 = np.array(f2[:]) # a numpy array of all of the training B features

	# create a mesh to plot the decision boundary in
	x_min, x_max = f1.min() - 1, f1.max() + 1
	y_min, y_max = f2.min() - 1, f2.max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.001), np.arange(y_min, y_max, 0.001))





	#C = np.arange(0.0008, 0.001, 0.000001)

	C = [.01, .8, .1, 1.0, 10.0, 100.0]

	accuracy = []
	# linear kernel 
	# iterate through values of C to find the one with the highest average accuracy across all k folds 
	for C in C:
		svc = svm.SVC(kernel='linear', C=float(C))
		svc.fit(X,y)
		# stratified, k-fold cross validation 
		# split the training data into k folds 
		cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)
		# use the other k-1 folds to train an svm and report accuracies for each run 
		scores = cross_val_score(svc,X,y,cv=cv)
		# record the average of the scores for each C value 
		mean = np.mean(scores)
		print mean
		accuracy.append([mean, C])


	max_a = 0
	best_c = -1
	# find the C value that yielded the model with the highest accuracy
	for i in xrange(len(accuracy)):
		if accuracy[i][0]>max_a:
			max_a = accuracy[i][0]
			best_c = accuracy[i][1]
	
	# print results  
	print "max accuracy for linear: " + str(max_a)
	print "best c parameter for linear: " + str(best_c)
	
	# train a svm with the chosen best parameters 
	svc = svm.SVC(kernel='linear',C=float(best_c)).fit(X, y)
	Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]) # make a prediction on the all possible values in a range to create decision boundary
	Z = Z.reshape(xx.shape)
	

	#print accuracy_score(y, Z)

	# plot the boundary based on all possible values in the range of xx and yy
	plt.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=0.8) # plot decision boundary 
	# plot training points 
	plt.plot(x0,y0, 'bs') # plot data points as red circles 
	plt.plot(x1,y1, 'g^') # plot data points as green triangles
	plt.xlabel('A')
	plt.ylabel('B')
	plt.title(titles[0])
	plt.savefig("5.pdf")
	plt.show()




	# rbf kernel 
	# iterate through values of gamma and C to find the one with the highest accuracy yield 
	accuracy2 = []
	gamma = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]
	C = [.01, .8, .1, 1.0, 10.0, 100.0]
	for g in gamma:
		for c in C:
			# create an rbf support vector machine 
			svc = svm.SVC(kernel='rbf', gamma=float(g), C=float(c))
			svc.fit(X,y)
			# same as in the linear case, perform k-fold splitting of the data 
			cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)
			# cross validation 
			scores = cross_val_score(svc,X,y,cv=cv)
			mean = np.mean(scores)
			print mean
			accuracy2.append([mean, g, c])

	max_a = 0
	best_g = -1
	best_c = -1
	# find the C value that yielded the model with the highest accuracy
	for i in xrange(len(accuracy2)):
		if accuracy2[i][0]>max_a:
			max_a = accuracy2[i][0]
			best_g = accuracy2[i][1]
			best_c = accuracy2[i][2]
	
	# print results 
	print "best accuracy for rbf kernel " + str(max_a)
	print "best gamma parameter for rbf " + str(best_g)
	print "best c parameter for rbf: " + str(best_c)
	
	# train a svm with the chosen best parameters 
	svc = svm.SVC(kernel='rbf', gamma=float(best_g), C=float(best_c)).fit(X, y)
	Z = svc.predict(np.c_[xx.ravel(), yy.ravel()]) # make a prediction on the all possible values in a range to create decision boundary
	Z = Z.reshape(xx.shape)
	# plot the boundary based on all possible values in the range of xx and yy
	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
	# plot training points 
	plt.plot(x0,y0, 'bs') # plot data points as red circles 
	plt.plot(x1,y1, 'g^') # plot data points as green triangles
	plt.xlabel('A')
	plt.ylabel('B')
	plt.title(titles[1])
	plt.savefig("6.pdf")
	plt.show()



	# polynomial kernel 
	degree = [2,3,4]
	C = [.01, .8, .1, 1.0, 10.0, 100.0]
	accuracy3 = []
	# iterate through different degrees and c values to find the ones with the highest accuracy yield 
	for deg in degree:
		for c in C:	
			# create a polynomial kernel svc 
			svc = svm.SVC(kernel='poly', degree=deg, C=c)
			svc.fit(X,y)
			# split data into k folds and perform cross validation 
			cv = KFold(X.shape[0], 5, shuffle=True, random_state=33)
			scores = cross_val_score(svc,X,y,cv=cv)
			mean = np.mean(scores)
			print mean
			accuracy3.append([mean, deg, c])

	max_a = 0
	best_deg = -1
	best_c = -1
	# find the C value that yielded the model with the highest accuracy
	for i in xrange(len(accuracy3)):
		if accuracy3[i][0]>max_a:
			max_a = accuracy3[i][0]
			best_deg = accuracy3[i][1]
			best_c = accuracy3[i][2]
	
	# print results  
	print "best accuracy for polynomial " + str(max_a)
	print "best degree for poly kernel " + str(best_deg)
	print "best c parameter for poly: " + str(best_c)

	# train a svm with the best parameters 
	poly_svc = svm.SVC(kernel='poly', degree=best_deg, C=float(best_c)).fit(X, y)
	Z = poly_svc.predict(np.c_[xx.ravel(), yy.ravel()])
	# put result in color plot 
	Z = Z.reshape(xx.shape)
	plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
	# plot training points 
	plt.plot(x0,y0, 'bs') # plot data points as red circles 
	plt.plot(x1,y1, 'g^') # plot data points as green triangles
	plt.xlabel('A')
	plt.ylabel('B')
	plt.title(titles[2])
	plt.savefig("7.pdf")
	plt.show()





for i in xrange(len(train)):
	f1.append(train[i][0])
	f2.append(train[i][1])

X = np.array([[train[i][0],train[i][1]] for i in xrange(len(train))])
y = np.array([train[i][2] for i in xrange(len(train))])


buildClassifier(X,y, x0, y0, x1, y1, f1, f2, train, test)







