#!/usr/bin/env python
# coding: utf-8

import config
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# METRIC UTILS

def per_sample_RMSE(y_true, y_pred, unknown):
	error = np.square(y_true-y_pred)													# PER CHANNEL ERROR
	mask = (y_true!=unknown).astype(np.float32)											# CREATE MASK
	error = error * mask																# MULTIPLY MASK
	error, mask = np.sum(error, axis=2), (np.sum(mask, axis=2)>0).astype(np.float32)	# PER INSTANCE ERROR
	error = np.sum(error)/np.sum(mask)													# MEAN OF INSTANCE ERROR
	error = np.sqrt(error)																# ROOT OF MEAN ERROR
	return error

def per_node_RMSE(y_true, y_pred, unknown):
	error = np.square(y_true-y_pred)													# PER SAMPLE ERROR
	mask = (y_true!=unknown).astype(np.float32)											# CREATE MASK
	error = error * mask																# MULTIPLY MASK
	error, mask = np.sum(error, axis=2), (np.sum(mask, axis=2)>0).astype(np.float32)	# PER INSTANCE ERROR
	error = np.sum(error, axis=1)/np.sum(mask, axis=1)									# PER LAKE ERROR
	error = np.sqrt(error)																# ROOT OF LAKE ERROR
	# error = np.mean(error)																# MEAN OF LAKE RMSE
	return error, np.mean(error)

def per_sample_R2(y_true, y_pred, unknown):
	mask = (y_true!=unknown)
	score = r2_score(y_true[mask], y_pred[mask])
	return score

def per_node_R2(y_true, y_pred, unknown):
	score = []
	for node in range(y_true.shape[0]):
		mask = (y_true[node]!=unknown)
		score.append(r2_score(y_true[node][mask], y_pred[node][mask]))
	score = np.array(score)
	# score = np.mean(score)
	return score, np.mean(score)

# DATA UTILS

def unstride_array(strided_data):
	shape = strided_data.shape
	data = config.unknown*np.ones((shape[0], (1+(shape[1]//2)), shape[2], shape[3]))
	data[:,:,config.stride:] = strided_data[:,::2][:,:,config.stride:]
	data[:,:,:config.stride] = np.concatenate((config.unknown*np.ones((shape[0],1,config.stride,shape[3])), strided_data[:,1::2][:,:,config.stride+1:]), axis=1)
	data = np.reshape(data, (shape[0], -1, shape[-1]))
	return data

# PLOT UTILS

def plot(dataset, basin, channel):
	fig = plt.figure(figsize=(24, 3))
	fontsize = 10
	plt.plot(dataset["data"][basin, :, channel])
	plt.xticks(range(0, dataset["date"].shape[0], config.window), dataset["date"][::config.window], fontsize=fontsize, rotation=90)
	plt.savefig('{}_{}_{}.png'.format("train", basin, channel), transparent=True, bbox_inches = 'tight', pad_inches = 0)
	plt.show()

def plot(dataset, basin, channel):
	fig = plt.figure(figsize=(24, 3))
	fontsize = 10
	plt.plot(dataset["data"][basin, :, channel])
	plt.yticks(color='w')
	plt.xticks(color='w')
	plt.savefig('{}_{}_{}.png'.format("train", basin, channel), transparent=True, bbox_inches = 'tight', pad_inches = 0)
	plt.show()