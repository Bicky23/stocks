# import modules
import os
import json
import pandas as pd
from unittest import TestCase
from inspect import getfullargspec
from stocks.cmd import train_model, ask_model
from stocks import DataPreparation, linearModel, dtModel

# load parameters from config.json
params = json.load(open("config.json"))
name = ["n", "o"]
train_size = params["train_size"]

# "n.csv"
path_n = params["datasources"][name[0]]
model_path_n = params["models"][name[0]]
dt_model = dtModel.deserialize(model_path_n)
processed_path_n = params["dest_path"][name[0]]
X_train_n = pd.read_csv(os.path.join(processed_path_n, "X_train.csv")).iloc[:,1:]
X_test_n = pd.read_csv(os.path.join(processed_path_n, "X_test.csv")).iloc[:,1:]
y_train_n = pd.read_csv(os.path.join(processed_path_n, "y_train.csv")).iloc[:,1:]
y_test_n = pd.read_csv(os.path.join(processed_path_n, "y_test.csv")).iloc[:,1:]
y_pred_n = dt_model.predict(X_test_n)

# "o.csv"
path_o = params["datasources"][name[1]]
model_path_o = params["models"][name[1]]
linear_model = linearModel.deserialize(model_path_o)
processed_path_o = params["dest_path"][name[1]]
X_train_o = pd.read_csv(os.path.join(processed_path_o, "X_train.csv")).iloc[:,1:]
X_test_o = pd.read_csv(os.path.join(processed_path_o, "X_test.csv")).iloc[:,1:]
y_train_o = pd.read_csv(os.path.join(processed_path_o, "y_train.csv")).iloc[:,1:]
y_test_o = pd.read_csv(os.path.join(processed_path_o, "y_test.csv")).iloc[:,1:]
y_pred_o = linear_model.predict(X_test_o)


class TestAll(TestCase):

	def test_len_args_train(self):
		arg = getfullargspec(train_model).args
		self.assertEqual(len(arg), 1, "Number of arguments should be 1. You have given {} arguments.".format(len(arg)))

	def test_len_args_ask(self):
		arg = getfullargspec(ask_model).args
		self.assertEqual(len(arg), 7, "Number of arguments should be 1. You have given {} arguments.".format(len(arg)))

	# "n.csv"
	def test_X_train_shape_n(self):
		self.assertEqual(X_train_n.shape, (420,13), "Shape of X_train is incorrect")

	def test_X_test_shape_n(self):
		self.assertEqual(X_test_n.shape, (105,13), "Shape of X_test is incorrect")

	def test_y_train_shape_n(self):
		self.assertEqual(y_train_n.shape, (420,1), "Shape of y_train is incorrect")

	def test_y_test_shape_n(self):
		self.assertEqual(y_test_n.shape, (105,1), "Shape of y_test is incorrect")

	def test_y_pred_val_n(self):
		self.assertAlmostEqual(y_pred_n[0], 0.014, 2, "First predicted value is incorrect")

	def test_y_pred_len_n(self):
		self.assertEqual(y_pred_n.shape, (105,), "Shape of y_pred is incorrect")

	# "o.csv"
	def test_X_train_shape_o(self):
		self.assertEqual(X_train_o.shape, (420,13), "Shape of X_train is incorrect")

	def test_X_test_shape_o(self):
		self.assertEqual(X_test_o.shape, (105,13), "Shape of X_test is incorrect")

	def test_y_train_shape_o(self):
		self.assertEqual(y_train_o.shape, (420,1), "Shape of y_train is incorrect")

	def test_y_test_shape_o(self):
		self.assertEqual(y_test_o.shape, (105,1), "Shape of y_test is incorrect")

	def test_y_pred_val_o(self):
		self.assertAlmostEqual(y_pred_o[0], -0.007, 2, "First predicted value is incorrect")

	def test_y_pred_len_o(self):
		self.assertEqual(y_pred_o.shape, (105,), "Shape of y_pred is incorrect")






