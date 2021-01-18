#!/usr/bin/env python

import glob

from tensorflow.keras.models import load_model

def gen_filename(pattern):
	filename_list = glob.glob(pattern)
	for filename in sorted(filename_list,key=epoch_from_filename):
		yield filename 


def epoch_from_filename(filename):
	epoch = int(filename.split("-")[1].split(".")[0])
	return epoch


def gen_loaded_model(filename_gen):
	for filename in filename_gen:
		model = load_model(filename)
		yield loaded_model
