# import modules
import os
import json
from unittest import TestCase
from inspect import getfullargspec
from stocks.cmd import train_model, ask_model
from stocks import DataPreparation, linearModel

# load parameters from config.json
params = json.load(open("config.json"))
name = "n"
path = params["datasources"][name]
train_size = params["train_size"]
train_model(name)
linear_model_path = params["models"][name]
linear_model = linearModel.deserialize(linear_model_path)


X_train, X_test, y_train, y_test = DataPreparation(path).feature_engineering(name, train_size)
y_pred = linear_model.predict(X_test)
print(y_pred[0])



class TestAll(TestCase):

	def test_len_args_train(self):
		arg = getfullargspec(train_model).args
		self.assertEqual(len(arg), 1, "Number of arguments should be 1. You have given {} arguments.".format(len(arg)))

	def test_len_args_ask(self):
		arg = getfullargspec(ask_model).args
		self.assertEqual(len(arg), 7, "Number of arguments should be 1. You have given {} arguments.".format(len(arg)))

	def test_X_train_shape(self):
		self.assertEqual(X_train.shape, (420,13), "Shape of X_train is incorrect")

	def test_X_test_shape(self):
		self.assertEqual(X_test.shape, (105,13), "Shape of X_test is incorrect")

	def test_y_train_shape(self):
		self.assertEqual(y_train.shape, (420,), "Shape of y_train is incorrect")

	def test_y_test_shape(self):
		self.assertEqual(y_test.shape, (105,), "Shape of y_test is incorrect")

	def test_y_pred_val(self):
		self.assertAlmostEqual(y_pred[0], 0.007, 2, "First predicted value is incorrect")

	def test_y_pred_len(self):
		self.assertEqual(y_pred.shape, (105,), "Shape of y_pred is incorrect")






