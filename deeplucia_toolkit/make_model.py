#!/usr/bin/env python

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import regularizers

from tensorflow.keras.models import Model

from tensorflow.keras.layers import concatenate , Conv2D , Dense , Dropout , Flatten , Input , MaxPooling2D , Reshape , ZeroPadding2D

#import kerastuner as kt

#from hyperopt import Trials, STATUS_OK, tpe
#from hyperas import optim
#from hyperas.distributions import choice, uniform



def seq_epi_cosmos_model(summary=True):
	fs_sequence = 40 # filter size
	padding_sequence = int(fs_sequence/2)
	fs_epigenome = 10 # filter size
	padding_epigenome = int(fs_epigenome/2)
	sequence_size = (5000, 4)
	epigenome_size = (200, 12)

	### Input setting
	sequence1_input = Input(shape=sequence_size, name='sequence1')
	sequence2_input = Input(shape=sequence_size, name='sequence2')
	sequence1_input_reshape = Reshape((5000, 4, 1))(sequence1_input)
	sequence2_input_reshape = Reshape((5000, 4, 1))(sequence2_input)
	sequence_path = concatenate([sequence1_input_reshape, sequence2_input_reshape], axis=-1)
	sequence_path = ZeroPadding2D(((padding_sequence, padding_sequence-1), (0,0)))(sequence_path)

	epigenome1_input = Input(shape=epigenome_size, name='epigenome1')
	epigenome2_input = Input(shape=epigenome_size, name='epigenome2')
	epigenome1_input_reshape = Reshape((200, 12, 1))(epigenome1_input)
	epigenome2_input_reshape = Reshape((200, 12, 1))(epigenome2_input)
	epigenome_path = concatenate([epigenome1_input_reshape, epigenome2_input_reshape], axis=-1)
	epigenome_path = ZeroPadding2D(((padding_epigenome, padding_epigenome-1), (0,0)))(epigenome_path)

	### PATH
	sequence_path = Conv2D(filters=512, kernel_size=(fs_sequence,4), padding='valid', activation='relu')(sequence_path)
	sequence_path = MaxPooling2D((50,1))(sequence_path)

	epigenome_path = Conv2D(filters=512, kernel_size=(fs_epigenome,12), padding='valid', activation='relu')(epigenome_path)
	epigenome_path = MaxPooling2D((2,1))(epigenome_path)

	### COMBINED PATH
	combined_path = concatenate([sequence_path, epigenome_path], axis=2)

	### new line
	combined_path = Dropout(0.5)(combined_path)
	combined_path = ZeroPadding2D(((5,4), (0,0)))(combined_path)
	combined_path = Conv2D(filters=2048, kernel_size=(10,2), padding='valid', activation='relu')(combined_path)
	combined_path = MaxPooling2D((100,1))(combined_path)
	combined_path = Flatten()(combined_path)
	combined_path = Dropout(0.5)(combined_path) #2019.12.19 initially 0.3
	combined_path = Dense(1024, activation='relu')(combined_path)

	combined_path = Dropout(0.5)(combined_path) #2019.12.19 initially 0.2
	combined_path = Dense(2, activation='softmax')(combined_path)

	model = Model(inputs=(sequence1_input, sequence2_input, epigenome1_input, epigenome2_input), outputs=combined_path)

	return model


def seq_epi_20200516_v1_model(summary=True):
	seq_dim = (5000,4)
	epi_dim = (200,12)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")

	seq_path_one = layers.Reshape((*seq_dim,1))(seq_input_one)
	seq_path_two = layers.Reshape((*seq_dim,1))(seq_input_two)
	epi_path_one = layers.Reshape((*epi_dim,1))(epi_input_one)
	epi_path_two = layers.Reshape((*epi_dim,1))(epi_input_two)

	seq_path = layers.concatenate([seq_path_one,seq_path_two],axis=-1)
	epi_path = layers.concatenate([epi_path_one,epi_path_two],axis=-1)

	seq_path = layers.Conv2D(filters=128, kernel_size=(40, 4), padding='same', activation='relu')(seq_path)
	epi_path = layers.Conv2D(filters=128, kernel_size=(10,12), padding='same', activation='relu')(epi_path)

	seq_path = layers.MaxPooling2D((50,1))(seq_path)
	epi_path = layers.MaxPooling2D(( 2,1))(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Conv2D(filters=512, kernel_size=(10,2), padding='same', activation='relu')(combined_path)
	combined_path = layers.MaxPooling2D((100,1))(combined_path)
	combined_path = layers.Flatten()(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(128,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(2,activation="softmax")(combined_path)

	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model 

def seq_epi_20200528_v1_model(summary=True):
	seq_dim = (5000,4)
	epi_dim = (200,12)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")

	seq_path_one = layers.Reshape((*seq_dim,1))(seq_input_one)
	seq_path_two = layers.Reshape((*seq_dim,1))(seq_input_two)
	epi_path_one = layers.Reshape((*epi_dim,1))(epi_input_one)
	epi_path_two = layers.Reshape((*epi_dim,1))(epi_input_two)

	seq_path = layers.concatenate([seq_path_one,seq_path_two],axis=-1)
	epi_path = layers.concatenate([epi_path_one,epi_path_two],axis=-1)

	seq_path = layers.Conv2D(filters=128, kernel_size=(40, 4), padding='same', activation='relu')(seq_path)
	epi_path = layers.Conv2D(filters=128, kernel_size=(10,12), padding='same', activation='relu')(epi_path)

	seq_path = layers.MaxPooling2D((50,1))(seq_path)
	epi_path = layers.MaxPooling2D(( 2,1))(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Conv2D(filters=512, kernel_size=(10,2), padding='same', activation='relu')(combined_path)
	combined_path = layers.MaxPooling2D((100,1))(combined_path)
	combined_path = layers.Flatten()(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(512,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(32,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)

	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model 


def seq_epi_20200901_v0_model(summary=True):
	seq_dim = (5000,4)
	epi_dim = (200,12)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")

	seq_path_one = layers.Reshape((*seq_dim,1))(seq_input_one)
	seq_path_two = layers.Reshape((*seq_dim,1))(seq_input_two)
	epi_path_one = layers.Reshape((*epi_dim,1))(epi_input_one)
	epi_path_two = layers.Reshape((*epi_dim,1))(epi_input_two)

	seq_path = layers.concatenate([seq_path_one,seq_path_two],axis=-1)
	epi_path = layers.concatenate([epi_path_one,epi_path_two],axis=-1)

	seq_path = layers.Conv2D(filters=512, kernel_size=(40, 4), padding='same', activation='relu')(seq_path)
	epi_path = layers.Conv2D(filters=512, kernel_size=(10,12), padding='same', activation='relu')(epi_path)

	seq_path = layers.MaxPooling2D((50,1))(seq_path)
	epi_path = layers.MaxPooling2D(( 2,1))(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Conv2D(filters=2048, kernel_size=(10,2), padding='same', activation='relu')(combined_path)
	combined_path = layers.MaxPooling2D((100,1))(combined_path)
	combined_path = layers.Flatten()(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(1024,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)

	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model


def seq_epi_20200901_v1_model(summary=True):
	seq_dim = (5000,4)
	epi_dim = (200,12)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")

	seq_path_one = layers.Reshape((*seq_dim,1))(seq_input_one)
	seq_path_two = layers.Reshape((*seq_dim,1))(seq_input_two)
	epi_path_one = layers.Reshape((*epi_dim,1))(epi_input_one)
	epi_path_two = layers.Reshape((*epi_dim,1))(epi_input_two)

	seq_path = layers.concatenate([seq_path_one,seq_path_two],axis=-1)
	epi_path = layers.concatenate([epi_path_one,epi_path_two],axis=-1)

	seq_path = layers.Conv2D(filters=128, kernel_size=(40, 4), padding='same', activation='relu')(seq_path)
	epi_path = layers.Conv2D(filters=128, kernel_size=(10,12), padding='same', activation='relu')(epi_path)

	seq_path = layers.MaxPooling2D((50,1))(seq_path)
	epi_path = layers.MaxPooling2D(( 2,1))(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Conv2D(filters=512, kernel_size=(10,2), padding='same', activation='relu')(combined_path)
	combined_path = layers.MaxPooling2D((100,1))(combined_path)
	combined_path = layers.Flatten()(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(512,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(64,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(8,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)

	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model 


def seq_epi_20200901_v2_model(summary=True):
	seq_dim = (5000,4)
	epi_dim = (200,12)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")

	seq_path_one = layers.Reshape((*seq_dim,1))(seq_input_one)
	seq_path_two = layers.Reshape((*seq_dim,1))(seq_input_two)
	epi_path_one = layers.Reshape((*epi_dim,1))(epi_input_one)
	epi_path_two = layers.Reshape((*epi_dim,1))(epi_input_two)

	seq_path = layers.concatenate([seq_path_one,seq_path_two],axis=-1)
	epi_path = layers.concatenate([epi_path_one,epi_path_two],axis=-1)

	seq_path = layers.Conv2D(filters=128, kernel_size=(40, 4), padding='same', activation='relu')(seq_path)
	epi_path = layers.Conv2D(filters=128, kernel_size=(10,12), padding='same', activation='relu')(epi_path)

	seq_path = layers.MaxPooling2D((50,1))(seq_path)
	epi_path = layers.MaxPooling2D(( 2,1))(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Conv2D(filters=32, kernel_size=(1,1), padding='same', activation='relu',name="bottleneck")(combined_path)
	combined_path = layers.Conv2D(filters=512, kernel_size=(10,2), padding='same', activation='relu')(combined_path)
	combined_path = layers.MaxPooling2D((100,1))(combined_path)
	combined_path = layers.Flatten()(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(512,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(64,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(8,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)

	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model 


def seq_epi_20201012_v1_model(summary=True):
	seq_dim = (5000,4)
	epi_dim = (200,12)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")

	seq_path = layers.concatenate([seq_input_one,seq_input_two],axis=1,name="seq_concat")
	epi_path = layers.concatenate([epi_input_one,epi_input_two],axis=1,name="epi_concat")

	seq_path = layers.Conv1D(filters=32, kernel_size=32, padding='same', activation='relu')(seq_path)
	epi_path = layers.Conv1D(filters=16, kernel_size=8, padding='same', activation='relu')(epi_path)

	seq_path = layers.MaxPooling1D(pool_size=30,strides=10,padding="same")(seq_path)
	seq_path = layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='relu')(seq_path)

	seq_path = layers.MaxPooling1D(pool_size=5,padding="same")(seq_path)
	epi_path = layers.MaxPooling1D(2)(epi_path)

	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Conv1D(filters=512,kernel_size=4)(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Flatten()(combined_path)
	combined_path = layers.Dense(256,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(16,activation="relu")(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model



"""
def seq_epi_hyperastunable_20201013_v1_model():

	activation_fun = {{choice(['relu', 'elu', 'selu'])}}
	seq_dim = (5000,4)
	epi_dim = (200,12)

	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")

	seq_path = layers.concatenate([seq_input_one,seq_input_two],axis=1,name="seq_concat")
	epi_path = layers.concatenate([epi_input_one,epi_input_two],axis=1,name="epi_concat")

	seq_path = layers.Conv1D(filters={{choice([16, 32, 64,128])}}, kernel_size=32, padding='same')(seq_path)
	seq_path = layers.Activation(activation_fun)(seq_path)
	seq_path = layers.MaxPooling1D(pool_size=10)(seq_path)
	seq_path = layers.Dropout({{uniform(0,1)}})(seq_path)

	seq_path = layers.Conv1D(filters={{choice([16, 32, 64,128])}}, kernel_size={{choice([2,4,8])}}, padding='same')(seq_path)
	seq_path = layers.Activation(activation_fun)(seq_path)
	seq_path = layers.MaxPooling1D(pool_size=5)(seq_path)
	seq_path = layers.Dropout({{uniform(0,1)}})(seq_path)


	epi_path = layers.Conv1D(filters={{choice([16, 32, 64,128])}}, kernel_size=8, padding='same')(epi_path)
	epi_path = layers.Activation(activation_fun)(epi_path)
	epi_path = layers.MaxPooling1D(pool_size=2)(epi_path)
	epi_path = layers.Dropout({{uniform(0,1)}})(epi_path)


	combined_path = layers.concatenate([seq_path, epi_path], axis=2)

	combined_path = layers.Conv1D(filters={{choice([128,256,512,1024])}},kernel_size={{choice([2,4,8])}})(combined_path)
	combined_path = layers.Activation(activation_fun)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size={{choice([1, 2, 4])}})(combined_path)
	combined_path = layers.Dropout({{uniform(0,1)}})(combined_path)

	combined_path = layers.Conv1D(filters={{choice([128,256,512,1024])}},kernel_size={{choice([2,4,8])}})(combined_path)
	combined_path = layers.Activation(activation_fun)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size={{choice([1, 2])}})(combined_path)
	combined_path = layers.Dropout({{uniform(0,1)}})(combined_path)

	combined_path = layers.Flatten()(combined_path)

	combined_path = layers.Dense({{choice([64,128,256,512,1024,2048])}})(combined_path)
	combined_path = layers.Activation(activation_fun)(combined_path)
	combined_path = layers.Dropout({{uniform(0,1)}})(combined_path)

	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model

def seq_epi_tunable_20201013_v1_model(hp):
	seq_dim = (5000,4)
	epi_dim = (200,12)

	hp_activation_fun = hp.Choice("activator", values = ['relu', 'elu', 'selu'])
	hp_dropout_ratio = hp.Float("dropout_ratio",min_value = 0,max_value=1)

	hp_num_chan_seq_one = hp.Choice("num_chan_seq_one", values = [16,32,64,128])
	hp_num_chan_seq_two = hp.Choice("num_chan_seq_two", values = [16,32,64,128])
	hp_num_chan_epi_one = hp.Choice("num_chan_epi_one", values = [16,32,64,128])

	hp_num_chan_comb_one = hp.Choice("num_chan_comb_one", values = [128,256,512,1024])
	hp_num_chan_comb_two = hp.Choice("num_chan_comb_two", values = [128,256,512,1024])


	hp_len_kern_seq_two = hp.Choice("num_chan_seq_two", values = [2,4,8])
	hp_len_kern_comb_one = hp.Choice("len_kern_comb_one", values = [2,4,8])
	hp_len_kern_comb_two = hp.Choice("len_kern_comb_two", values = [2,4,8])

	hp_pool_size_comb_one = hp.Choice("pool_size_comb_one", values = [1,2,4])
	hp_pool_size_comb_two = hp.Choice("pool_size_comb_two", values = [1,2])

	hp_dense_unit = hp.Choice("dense_unit", values = [64,128,256,512,1024,2048])
	
	hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])


	seq_input_one = layers.Input(shape=seq_dim,name="seq_feature_one")
	seq_input_two = layers.Input(shape=seq_dim,name="seq_feature_two")
	epi_input_one = layers.Input(shape=epi_dim,name="epi_feature_one")
	epi_input_two = layers.Input(shape=epi_dim,name="epi_feature_two")

	seq_path = layers.concatenate([seq_input_one,seq_input_two],axis=1,name="seq_concat")
	epi_path = layers.concatenate([epi_input_one,epi_input_two],axis=1,name="epi_concat")

	seq_path = layers.Conv1D(filters=hp_num_chan_seq_one, kernel_size=32, padding='same')(seq_path)
	seq_path = layers.Activation(hp_activation_fun)(seq_path)
	seq_path = layers.MaxPooling1D(pool_size=10)(seq_path)
	seq_path = layers.Dropout(hp_dropout_ratio)(seq_path)

	seq_path = layers.Conv1D(filters=hp_num_chan_seq_two, kernel_size=hp_len_kern_seq_two, padding='same')(seq_path)
	seq_path = layers.Activation(hp_activation_fun)(seq_path)
	seq_path = layers.MaxPooling1D(pool_size=5)(seq_path)
	seq_path = layers.Dropout(hp_dropout_ratio)(seq_path)


	epi_path = layers.Conv1D(filters=hp_num_chan_epi_one, kernel_size=8, padding='same')(epi_path)
	epi_path = layers.Activation(hp_activation_fun)(epi_path)
	epi_path = layers.MaxPooling1D(pool_size=2)(epi_path)
	epi_path = layers.Dropout(hp_dropout_ratio)(epi_path)


	combined_path = layers.concatenate([seq_path, epi_path], axis=2)

	combined_path = layers.Conv1D(filters=hp_num_chan_comb_one,kernel_size=hp_len_kern_comb_one)(combined_path)
	combined_path = layers.Activation(hp_activation_fun)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=hp_pool_size_comb_one)(combined_path)
	combined_path = layers.Dropout(hp_dropout_ratio)(combined_path)

	combined_path = layers.Conv1D(filters=hp_num_chan_comb_one,kernel_size=hp_len_kern_comb_two)(combined_path)
	combined_path = layers.Activation(hp_activation_fun)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=hp_pool_size_comb_two)(combined_path)
	combined_path = layers.Dropout(hp_dropout_ratio)(combined_path)

	combined_path = layers.Flatten()(combined_path)

	combined_path = layers.Dense(hp_dense_unit)(combined_path)
	combined_path = layers.Activation(hp_activation_fun)(combined_path)
	combined_path = layers.Dropout(hp_dropout_ratio)(combined_path)

	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)
	model.compile(loss='binary_crossentropy',metrics=["accuracy", "Precision" , "Recall", "AUC"],optimizer=keras.optimizers.Adam(lr=hp_learning_rate))

	return model
"""

def seq_epi_20201015_v1_model(summary=True):
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

	seq_path = layers.MaxPooling1D(pool_size=30,strides=10,padding="same")(seq_path)
	seq_path = layers.Conv1D(filters=32, kernel_size=4, padding='same', activation='relu')(seq_path)

	seq_path = layers.MaxPooling1D(pool_size=5,padding="same")(seq_path)
	epi_path = layers.MaxPooling1D(2)(epi_path)
	combined_path = layers.concatenate([seq_path, epi_path], axis=2)
	combined_path = layers.Conv1D(filters=32, kernel_size=1, padding='same', activation='relu',name="bottleneck")(combined_path)
	combined_path = layers.Conv1D(filters=256 ,kernel_size=10)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=10,padding="same")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Flatten()(combined_path)
	combined_path = layers.Dense(1024,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(64,activation=tf.nn.swish)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model


def seq_epi_20201015_v2_model(summary=True):
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
	combined_path = layers.Conv1D(filters=512 ,kernel_size=10)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=100,padding="same")(combined_path)
	combined_path = layers.Flatten()(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(512,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(64,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(8,activation="relu")(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model


	
def seq_epi_20201015_v3_model(summary=True):
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
	combined_path = layers.Conv1D(filters=512 ,kernel_size=10)(combined_path)
	combined_path = layers.MaxPooling1D(pool_size=100,padding="same")(combined_path)
	combined_path = layers.Flatten()(combined_path)
	combined_path = layers.Dropout(0.5)(combined_path)
	combined_path = layers.Dense(512,activation="relu")(combined_path)
	combined_path = layers.Dense(1,activation="sigmoid")(combined_path)
	
	model = Model(inputs=(seq_input_one,seq_input_two,epi_input_one,epi_input_two),outputs=combined_path)

	return model


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
