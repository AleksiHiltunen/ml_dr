from tensorflow import keras
import tensorflow as tf
import numpy as np
import random
import math

TRAIN_SET_SIZE = 100000
TEST_SET_SIZE = 10000

DATA_MIN = -50
DATA_MAX = 50

"""Generates random datasets for polynomy function"""
def create_dataset():
	train_set = create_set(TRAIN_SET_SIZE)
	test_set = create_set(TEST_SET_SIZE)
	train_set_x = [] 
	train_set_y = [] 
	test_set_x = [] 
	test_set_y = []
	for item in train_set:
		train_set_x.append(item[0])
		train_set_y.append(item[1])
		
	for item in test_set:
		test_set_x.append(item[0])
		test_set_y.append(item[1])
	
	train_x = np.array(train_set_x, dtype="float16", ndmin=2)
	train_y = np.array(train_set_y, dtype="float16", ndmin=2)
	test_x = np.array(test_set_x, dtype="float16", ndmin=2)
	test_y = np.array(test_set_y, dtype="float16", ndmin=2)

	return (train_x, train_y), (test_x, test_y)
	
def create_set(size):
	dset = []
	while len(dset) < size:
		a = random.randint(DATA_MIN, DATA_MAX)
		b = random.randint(DATA_MIN, DATA_MAX)
		c = random.randint(DATA_MIN, DATA_MAX)
		input = [a,b,c]
		output = calc_polynom(a,b,c)
		if(output == -1):
			continue
		dset.append(output)
	
	return dset
	
def calc_polynom(a,b,c):
	ans1 = None
	ans2 = None
	if a == 0:
		return -1
	if (b*b-a*c*4) < 0:
		return -1
		
	"""======================="""
	ans1 = (-b-math.sqrt(b*b-a*c*4))/(2*a)
	ans2 = (-b+math.sqrt(b*b-a*c*4))/(2*a)
	"""======================="""
	
	return [[a,b,c],[ans1, ans2]]
	
(train_x, train_y),(test_x, test_y) = create_dataset()

"""Alright now we have a large number of train and test data"""
"""Learning starts here"""

model = keras.Sequential([	
							keras.layers.Dense(3), 
							keras.layers.Dense(100, activation=tf.nn.relu),
							keras.layers.Dense(100, activation=tf.nn.relu),
							keras.layers.Dense(2)
						])
						
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

"""Model is now initiated but not taught"""
"""print("Model is now initiated but not taught. Making a prediction should return very random values")
input("Press Enter to continue...")
index = random.randint(0, len(test_x)-1)
predictions = model.predict(test_x)
p = predictions[index]
print(p)

input("Press Enter to continue...")
"""
print("=================BEGIN TRAINING============================")
history = model.fit(train_x, train_y, epochs=50)
loss, acc = model.evaluate(test_x, test_y)	
print("Model accuracy is:", acc)
print("===================END TRAINING============================")

print("Model is taught now and should return reasonable values")
input("Press Enter to continue...")
while True:
	index = random.randint(0, len(test_x)-1)
	predictions = model.predict(test_x)
	p = predictions[index]
	print("Predicted roots:", p)
	print("Actual roots:", test_y[index])
	i = input("Press Enter to continue...")
	if(i == "x"):
		break
