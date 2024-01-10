#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

DATASET = "camels_GB"
EXPERIMENT = "baseline"

# FILES INFO
DATA_DIR = os.path.join("..", "..", "DATA")
RAW_DIR = os.path.join(DATA_DIR, "{}".format(DATASET), "RAW")
PREPROCESSED_DIR = os.path.join(DATA_DIR, "{}".format(DATASET), "PREPROCESSED")
RESULT_DIR = os.path.join(DATA_DIR, "{}".format(DATASET), "{}".format(EXPERIMENT), "RESULT")
MODEL_DIR = os.path.join(DATA_DIR, "{}".format(DATASET), "{}".format(EXPERIMENT), "MODEL")

if not os.path.exists(PREPROCESSED_DIR):
	os.makedirs(PREPROCESSED_DIR)
if not os.path.exists(RESULT_DIR):
	os.makedirs(RESULT_DIR)
if not os.path.exists(MODEL_DIR):
	os.makedirs(MODEL_DIR)

# TIME SERIES INFO
train_year = {"start":1999, "end":2009}
valid_year = {"start":1997, "end":1999}
test_year = {"start":1989, "end":1997}
window = 365
stride = window//2

# CHANNELS INFO
channels_names = np.array([
	'area','elev_mean','dpsbar','sand_perc','silt_perc','clay_perc','porosity_hypres','conductivity_hypres',			#Static Features
	'soil_depth_pelletier','dwood_perc','ewood_perc','crop_perc','urban_perc','reservoir_cap','p_mean','pet_mean',		#Static Features
	'p_seasonality','frac_snow','high_prec_freq','low_prec_freq','high_prec_dur','low_prec_dur',						#Static Features
	'precipitation','peti','temperature',																				#Dynamic Features
	'discharge_spec'																									#StreamFlow
])
channels = list(range(len(channels_names)))
static_channels = channels[:22]
dynamic_channels = channels[22:25]
output_channels = [channels[-1]]

# LABELS INFO
add = 0.005
unknown = -999

# MODEL INFO
code_dim = 256
device = "cuda"

# TRAIN INFO
train = True
batch_size = 200
epochs = 300
learning_rate = 3e-4