import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.datasets import mnist

class NN_Model:
	def __init__(self):
		try:
			self.mnist = keras.datasets.mnist
			self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
			self.predictions = None
			random.seed(1)
		except Exception as e:
			print("NN_Model initiation error:", e)
			
	def load_dataset(self):
		try:
			(self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
		except Exception as e:
			print("Data load error:", e)
			
	def init_model(self, hidden_nodes=128):	
		self.model = keras.Sequential([
										keras.layers.Flatten(input_shape=(28,28)), 
										keras.layers.Dense(hidden_nodes, activation=tf.nn.relu),
										keras.layers.Dense(10, activation=tf.nn.softmax)
									])
		self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
		
	def train_model(self, epochs=10):
		self.history = self.model.fit(self.x_train, self.y_train, epochs=epochs, validation_split=0.25, batch_size=32, verbose=1)
		self.predictions = self.model.predict(self.x_test)
		loss, acc = self.model.evaluate(self.x_test, self.y_test)	
		print("Model accuracy is:", acc)
		
	def predict(self, index):
		try:
			p = self.predictions[index]
			val = 0
			for i in range(10):
				if p[i] == 1.0:
					val = i
					break
					
			print("Model predicted a value:", val)
			self.plot_test_image(self.x_test[index])
			
		except Exception as e:
			print("Predict error:", e)
			
	def predict_single(self, data):
		p = self.model.predict(data)
		val = 0
		for i in range(10):
			if p[0][i] == 1.0:
				val = i
				break
		print("Model predicted a value:", val)
		self.plot_test_image(data[0])
	
	def plot_test_image(self, data):
		plt.figure()
		plt.imshow(data)
		plt.grid(False)
		plt.show()
		
	def plot_train_image(self, data):
		plt.figure()
		plt.imshow(data)
		plt.grid(False)
		plt.show()
		
	def plot_train_data(self):
		plt.plot(self.history.history['acc'])
		plt.plot(self.history.history['val_acc'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Test'], loc='upper left')
		plt.show()