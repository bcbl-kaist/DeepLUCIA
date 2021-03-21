#!/usr/bin/env python

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import regularizers

from tensorflow.keras.models import Model

from tensorflow.keras.layers import concatenate , Conv2D , Dense , Dropout , Flatten , Input , MaxPooling2D , Reshape , ZeroPadding2D


def seq_epi_20210105_v1_model(summary=True):
	seq_dim = (5000,4)
	epi_dim = (200,12)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")

	seq_path = layers.concatenate([seq_input_one,seq_input_two],axis=1,name="seq_concat")
	epi_path = layers.concatenate([epi_input_one,epi_input_two],axis=1,name="epi_concat")

	seq_path = layers.Conv1D(filters=128, kernel_size=40, padding='same', activation='relu')(seq_path)
	epi_path = layers.Conv1D(filters=128, kernel_size=10, padding='same', activation='relu')(epi_path)

	seq_path = layers.MaxPooling1D(50)(seq_path)
	epi_path = layers.MaxPooling1D(2)(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	
	combined_path = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation='relu',name="bottleneck")(combined_path)
	combined_path = layers.Conv1D(filters=512 ,kernel_size=10, padding='same',activation='relu')(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=100,padding="same")(combined_path)
	combined_path = layers.Flatten()(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(512,activation="relu")(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model