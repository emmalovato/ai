# Emma Peterson
# edp2117
# AI Homework 3 
# Problem 1: Regression


# Part 1: Linear Regression with One Feature

from numpy import linalg
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import math
import pdb


def part_one():

	# load dataset

	f = open('girls_train.csv', 'r')
	text = f.read()
	# make a two dimensional array of (age, height) pairs
	data = text.split("\r\n")[:]
	for i in xrange(len(data)):
		data[i] = data[i].split(",")[:]
	for i in xrange(len(data)):
		data[i] = [float(data[i][0]),float(data[i][1])]


	# plot distribution of the data 

	x = np.array([data[i][0] for i in xrange(len(data))]) # list of ages
	y = np.array([data[j][1] for j in xrange(len(data))]) # list of heights
	plt.figure(1)
	plt.plot(x,y, 'ro') # plot data points as red circles 
	plt.ylabel('height')
	plt.xlabel('age')
	plt.title('Girls age vs height')
	plt.axis([1.5,9,0.8,1.5]) # adjusted axes


	# error function without data parameter, used for the 3D plot
	def loss(beta0, beta1): 
		totalError = 0.0
		# calculate the mean squared error for two weights 
		for i in xrange(len(data)):
			totalError += (y[i] - (beta1*x[i] + beta0))**2 
		return totalError/float(len(x))


	iterations = 1500
	# start at a point (0,0) in the (w0, w1) plane
	[beta0, beta1] = [0.0,0.0]
	weights = []
	errors = []
	# the learning rate 
	alpha = .05

	for i in xrange(iterations):
		# calculate the new weights based on the old ones and add them to the list
		weights.append(gradDescent(beta0, beta1, x, y, alpha))
		beta0 = weights[i][0]
		beta1 = weights[i][1]
		# calculate loss for the pair of weights 
		errors.append(loss(beta0, beta1))

	# find values of beta with the least error
	# the smallest error is the mean square error of the regression model on the training set 
	error_min = errors[0]
	pos = 0
	for j in xrange(len(errors)):
		if errors[j] < error_min:
			error_min = errors[j]
			pos = j
	print "The mean square error of the regression model on the training set is: "
	print error_min

	# plot the regression line on top of data points
	b = weights[pos][0]
	m = weights[pos][1]
	print "The weights beta0 and beta1 (the y-intercept and slope of the regression line) are: "
	print [b, m]
	x2 = np.array(x[:])
	y2 = np.array([m*x2[i]+b for i in xrange(len(x))])
	# try to make this line extend to the y-axis
	plt.plot(x2,y2)

	plt.savefig("1.pdf")
	plt.show()

	# plot the bowl-shaped cost function with respect to beta1 and beta2
	fig = plt.figure(2)
	ax = fig.gca(projection='3d')
	x3 = np.arange(-5,5,.01)
	y3 = np.arange(-5,5,.01)
	X,Y = np.meshgrid(x3, y3)
	Z = loss(X,Y)
	ax.set_xlabel("Weight 0")
	ax.set_ylabel("Weight 1")
	ax.set_zlabel("Loss")
	ax.set_title("Loss Function for Various Weight Vectors")
	ax.plot_surface(X,Y,Z)

	plt.savefig("2.pdf")
	plt.show()

	# using model to make a prediction 
	print "Height prediction for a 4.5 year old girl: "
	prediction = m*4.5 + b
	print prediction

	# Compute the mean square error for a test set of 20 girls
	f = open('girls_test.csv', 'r')
	text = f.read()
	# make a two dimensional array of (age, height) pairs
	test = text.split("\r\n")[:]
	for i in xrange(len(test)):
		test[i] = test[i].split(",")[:]
	for i in xrange(len(test)):
		test[i] = [float(test[i][0]),float(test[i][1])]
	mse = loss1(b, m, test)
	print "Mean squared error for test set: "
	print mse




# Implement gradient descent to find the beta's of the model
# beta1 refers to the slope of the line, beta0 refers to its y-intercept

# gradient descent function 
def gradDescent(b0_current, b1_current, x, y, alpha):
	b0_gradient = 0.0
	b1_gradient = 0.0
	N = float(len(x))
	# calculate gradients for each weight 
	for i in xrange(0, len(x)):
		b0_gradient += (-1/N)*(y[i] - (b1_current*x[i] + b0_current))
		b1_gradient += (-1/N)*x[i]*(y[i] - (b1_current*x[i] + b0_current))
	# step downhill!
	b0_new = b0_current - (alpha*b0_gradient)
	b1_new = b1_current - (alpha*b1_gradient)
	return [b0_new, b1_new]

# error function for everything but the 3D plot
def loss1(beta0, beta1, data): 
	x = [data[i][0] for i in xrange(len(data))]
	y = [data[i][1] for i in xrange(len(data))] 
	totalError = 0.0
	# calculate the mean squared error for two weights 
	for i in xrange(len(data)):
		totalError += (y[i] - (beta1*x[i] + beta0))**2 
	return totalError/float(len(x))




# Part 2: Linear Regression with Multiple Features



def part_two():

	# load dataset

	f = open('girls_age_weight_height_2_8.csv', 'r')
	text = f.read()
	# make a two dimensional array of (age, weight, height) 
	data = text.split("\r\n")[:]
	for i in xrange(len(data)):
		data[i] = data[i].split(",")[:]
	for i in xrange(len(data)):
		data[i] = [float(data[i][0]),float(data[i][1]), float(data[i][2])]

	# mean of age, weight 
	sum_age = 0.0
	for i in xrange(len(data)):
		sum_age += data[i][0]
	mean_age = sum_age/float(len(data))
	print "mean of age: " + str(mean_age)

	sum_weight = 0.0
	for i in xrange(len(data)):
		sum_weight += data[i][1]
	mean_weight = sum_weight/float(len(data))
	print "mean of weight: " + str(mean_weight)


	# standard deviation of age, weight 
	total = 0.0
	for i in xrange(len(data)):
		total += (data[i][0] - mean_age)**2
	total = total/(float(len(data) - 1))
	sd_age = math.sqrt(total)
	print "standard deviation of age: " + str(sd_age)

	total = 0.0
	for i in xrange(len(data)):
		total += (data[i][1] - mean_weight)**2
	total = total/(float(len(data) - 1))
	sd_weight = math.sqrt(total)
	print "standard deviation of weight: " + str(sd_weight)


	# scale each feature by its SD and set its mean to 0
	ages2 = []
	weights2 = []
	y2 = [data[i][2] for i in xrange(len(data))] # height is the classifier and doesn't need to be scaled
	for i in xrange(len(data)):
		x1 = data[i][0]
		x2 = data[i][1]
		x1 = (x1 - mean_age)/sd_age # scale age
		x2 = (x2 - mean_weight)/sd_weight # scale weight 
		ages2.append(x1)
		weights2.append(x2)


	iterations = 50
	# start at a point (0,0) in the (w0, w1) plane
	[a_beta0, a_beta1, a_beta2] = [0.0,0.0,0.0]
	result = []
	betas = []
	risks = []
	# the learning rate 
	alphas = [0.001,0.005,0.05,0.1,0.5,1]
	plt.figure(num=1, figsize=(12,8))

	# the x axis is always number of iterations (from 0 to 50)
	a_x = np.arange(0,50,1)

	for alpha in alphas:

		# each alpha gets a new array of 0's to fill with y values 
		a_y = np.zeros(50)

		for i in xrange(0,iterations):
			# calculate the new weights based on the old ones 
			result = gradDescent2(a_beta0, a_beta1, a_beta2, alpha, ages2, weights2, y2)
			a_beta0 = result[0]
			a_beta1 = result[1]
			a_beta2 = result[2]
			# calculate loss for the pair of weights 
			a_risk = risk(a_beta0, a_beta1, a_beta2, ages2, weights2, y2)
			# there is a calculated risk value for each iteration 
			a_y[i] = a_risk
			if alpha == 0.05: # this is the best alpha value according to the risk graph, so we need to save only its risks and betas
				risks.append(a_risk)
				betas.append([a_beta0, a_beta1, a_beta2])
		# plot a line for each value of alpha 
		plt.plot(a_x,a_y)
			

	
	plt.xlabel("Number of Iterations")
	plt.ylabel("Calculated Risk")
	plt.title("Risk Function for Various Values of Alpha")
	# put a legend below plot
	plt.legend(['a = .001', 'a = .005', 'a = .05', 'a = .1', 'a = .5', 'a = 1'], loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)
	plt.savefig("3.pdf")
	plt.show()

	# only print the beta values for the best alpha (0.05)
	risk_min = risks[0]
	pos = 0
	for j in xrange(len(risks)):
		if risks[j] < risk_min:
			risks_min = risks[j]
			pos = j
	[w0, w1, w2] = betas[j]
	print "The weights beta0, beta1, and beta2 for a learning rate of 0.05 are: "
	print [w0, w1, w2]
	
	# make a height prediction using these beta values for a 5 year old girl weighing 20 kilos
	# scale x values first
	p_x1 = (5.0 - mean_age)/sd_age # scale age
	p_x2 = (20.0 - mean_weight)/sd_weight # scale weight 
	prediction = p_x1*w1 + p_x2*w2 + w0
	print "Using gradient descent, the predicted height for a 5 year old girl weighing 20 kilos is: "
	print str(prediction) + " meters."



	# calculate betas with normal equation
	N = 2 # number of features
	X = [] # data matrix (matrix of inputs with one n-dimensional example per row)
	y = [] # the vector of outputs for the training examples
	w = [] # the vector of weights that minimizes loss
	for i in xrange(len(ages2)):
		X.append([1, ages2[i], weights2[i]])
	for i in xrange(len(y2)):
		y.append([y2[i]])
	X = np.array(X[:])
	y = np.array(y[:])
	transpose = np.transpose(X)
	product = transpose.dot(X)
	inverse = np.linalg.inv(product)
	prod = inverse.dot(transpose)
	w = prod.dot(y)
	print "The beta vector obtained from the normal equation is: "
	print w
	# use the betas to predict the height of a 5 year old girl weighing 20 kilos
	# scale x values first
	prediction = p_x1*w[1] + p_x2*w[2] + w[0]
	print "Using the normal equation, the predicted height for a 5 year old girl weighing 20 kilos is: "
	print str(prediction) + " meters."


# gradient descent with two features (age and weight are the features, height is the label)
def gradDescent2(b0, b1, b2, alpha, x1, x2, y):
	#pdb.set_trace()
	b0_gradient = 0.0
	b1_gradient = 0.0
	b2_gradient = 0.0
	N = float(len(x1))
	# calculate gradients for each weight 
	for i in xrange(len(x1)):
		b0_gradient += (1/N)*((b1*x1[i] + b2*x2[i] + b0) - y[i])
		b1_gradient += (1/N)*x1[i]*((b1*x1[i] + b2*x2[i] + b0) - y[i])
		b2_gradient += (1/N)*x2[i]*((b1*x1[i] + b2*x2[i] + b0) - y[i])
	# step downhill!
	b0 = b0 - (alpha*b0_gradient)
	b1 = b1 - (alpha*b1_gradient)
	b2 = b2 - (alpha*b2_gradient)
	return [b0, b1, b2]

# error function
def risk(beta0, beta1, beta2, ages, weights, y): 
	totalError = 0.0
	# calculate the mean squared error for two weights 
	for i in xrange(len(ages)):
		totalError += (y[i] - (beta1*ages[i] + beta2*weights[i] + beta0))**2 
	return totalError/float(2*len(ages))




part_one()
part_two()




