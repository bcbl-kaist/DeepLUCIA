### keras import 
import sys
import keras
import keras.backend as K
import tensorflow as tf
from keras.constraints import NonNeg
from keras import regularizers
from keras import initializers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, concatenate, Merge, Input, Reshape, BatchNormalization, Activation, Conv2D, MaxPooling2D, MaxPooling1D, Conv1D, ZeroPadding2D, LSTM, Add, Conv2DTranspose, TimeDistributed, CuDNNLSTM, Bidirectional, Concatenate

#### data size ####
NUM_CLASSES = 2
SEQ_LENGTH = 5000
EPIGENOME_LENGTH = 200
ONEHOT = 4

def shared_layer(feature_size=1):
	seq_size = (SEQ_LENGTH, ONEHOT)
	if feature_size:
		epigenome_size = (EPIGENOME_LENGTH, feature_size)

	### Input setting
	seq1_input	= Input(shape=seq_size, name='seq1')
	seq2_input	= Input(shape=seq_size, name='seq2')
	if feature_size:
		epigenome1_input	= Input(shape=epigenome_size, name='epigenome1')
		epigenome2_input	= Input(shape=epigenome_size, name='epigenome2')
	
	### SEQ PATH
	seq1_path = Conv1D(filters=256, kernel_size=40, padding='same', activation='relu')(seq1_input)
	seq1_path = MaxPooling1D(50)(seq1_path)
	seq2_path = Conv1D(filters=256, kernel_size=40, padding='same', activation='relu')(seq2_input)
	seq2_path = MaxPooling1D(50)(seq2_path)
	seq_path = concatenate([seq1_path, seq2_path], axis=2)

	### EPIGENOME PATH 2019.10.03 every epigenomic information such as DNaase, CTCF, DNAmeth are combined into single path
	if feature_size:
		epigenome1_path = Conv1D(filters=256, kernel_size=20, padding='same', activation='relu')(epigenome1_input)
		epigenome1_path = MaxPooling1D(2)(epigenome1_path)
		epigenome2_path = Conv1D(filters=256, kernel_size=20, padding='same', activation='relu')(epigenome2_input)
		epigenome2_path = MaxPooling1D(2)(epigenome2_path)
		epigenome_path = concatenate([epigenome1_path, epigenome2_path], axis=2)

	### COMBINED PATH
	if feature_size: #including epigenomic information
		combined_path = concatenate([seq_path, epigenome_path], axis=-1)
		combined_path = BatchNormalization()(combined_path)
		combined_path = Conv1D(filters=1024, kernel_size=10, padding='same', activation='relu')(combined_path)
		combined_path = MaxPooling1D(2)(combined_path)
	else: #only sequence
		combined_path = seq_path
		combined_path = Conv1D(filters=512, kernel_size=10, padding='same', activation='relu')(combined_path)
		combined_path = MaxPooling1D(2)(combined_path)

	combined_path = Flatten()(combined_path)
	combined_path = Dropout(0.3)(combined_path)

	if feature_size:
		return seq1_input, seq2_input, epigenome1_input, epigenome2_input, combined_path
	else:
		return seq1_input, seq2_input, combined_path

def specific_layer(combined_path, feature_size=1):
	if feature_size:
		combined_path = Dense(256, activation='relu')(combined_path)
	else:
		combined_path = Dense(128, activation='relu')(combined_path)

	combined_path = Dropout(0.2)(combined_path)
	combined_path = Dense(NUM_CLASSES, activation='softmax')(combined_path)

	return combined_path

def model5_multitaskModel(input_cellType, optimizer, feature_size=1, summary=True):
	cellType_list = ['E017','E116','E117','E119','E122','E123','E127']
	cellType_list.remove(input_cellType)

	### share layer construction
	if feature_size:
		seq1_input, seq2_input, epigenome1_input, epigenome2_input, multitask_shared_LAYER = shared_layer(feature_size=feature_size)
	else:
		seq1_input, seq2_input, multitask_shared_LAYER = shared_layer(feature_size=feature_size)

	### model dictionary for each cellType
	model_dict = dict()
	### specific layer for each cellType
	for cellType in cellType_list:
		specific_LAYER = specific_layer(multitask_shared_LAYER, feature_size=feature_size)
		if feature_size: #including epigenomic information
			model = Model(inputs=(seq1_input, seq2_input, epigenome1_input, epigenome2_input), outputs=specific_LAYER)
		else:
			model = Model(inputs=(seq1_input, seq2_input), outputs=specific_LAYER)

		model.compile(
			loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy', auc_roc]
		)
		model_dict[cellType] = model	

	if summary:
		for key, single_model in model_dict.iteritems():
			print('###', key, single_model)
			print('###Checking for shared layer MEMORY ADDRESS', single_model.get_layer('epigenome1'))
			single_model.summary()

	return model_dict
	
def model6_unifiedConvolution(feature_size=1, summary=True):
	fs_sequence = 40 # filter size
	padding_sequence = fs_sequence/2
	fs_epigenome = 10 # filter size
	padding_epigenome = fs_epigenome/2
	sequence_size = (SEQ_LENGTH, ONEHOT)
	if feature_size:
		epigenome_size = (EPIGENOME_LENGTH, feature_size)

	### Input setting
	sequence1_input	= Input(shape=sequence_size, name='sequence1')
	sequence2_input	= Input(shape=sequence_size, name='sequence2')
	sequence1_input_reshape = Reshape((SEQ_LENGTH, ONEHOT, 1))(sequence1_input)
	sequence2_input_reshape = Reshape((SEQ_LENGTH, ONEHOT, 1))(sequence2_input)
	sequence_path = concatenate([sequence1_input_reshape, sequence2_input_reshape], axis=-1)
	sequence_path = ZeroPadding2D(((padding_sequence, padding_sequence-1), (0,0)))(sequence_path)
	if feature_size:
		epigenome1_input	= Input(shape=epigenome_size, name='epigenome1')
		epigenome2_input	= Input(shape=epigenome_size, name='epigenome2')
		epigenome1_input_reshape = Reshape((EPIGENOME_LENGTH, feature_size, 1))(epigenome1_input)
		epigenome2_input_reshape = Reshape((EPIGENOME_LENGTH, feature_size, 1))(epigenome2_input)
		epigenome_path = concatenate([epigenome1_input_reshape, epigenome2_input_reshape], axis=-1)
		epigenome_path = ZeroPadding2D(((padding_epigenome, padding_epigenome-1), (0,0)))(epigenome_path)

	### SEQ PATH
	sequence_path = Conv2D(filters=512, kernel_size=(fs_sequence,4), padding='valid', activation='relu')(sequence_path)
	sequence_path = MaxPooling2D((50,1))(sequence_path)

	### EPIGENOME PATH 2019.10.03 every epigenomic information such as DNaase, CTCF, DNAmeth are combined into single path
	if feature_size:
		epigenome_path = Conv2D(filters=512, kernel_size=(fs_epigenome,feature_size), padding='valid', activation='relu')(epigenome_path)
		epigenome_path = MaxPooling2D((2,1))(epigenome_path)

	### COMBINED PATH
	if feature_size: #including epigenomic information
		combined_path = concatenate([sequence_path, epigenome_path], axis=2)
		### new line
		combined_path = Dropout(0.5)(combined_path)
		combined_path = ZeroPadding2D(((5,4), (0,0)))(combined_path)
		combined_path = Conv2D(filters=2048, kernel_size=(10,2), padding='valid', activation='relu')(combined_path)
		combined_path = MaxPooling2D((100,1))(combined_path)
		combined_path = Flatten()(combined_path)
		combined_path = Dropout(0.5)(combined_path) #2019.12.19 initially 0.3
		combined_path = Dense(1024, activation='relu')(combined_path)
	else: #only sequence
		combined_path = sequence_path
		combined_path = Conv2D(filters=512, kernel_size=(10,1), padding='valid', activation='relu')(combined_path)
		combined_path = MaxPooling2D((2,1))(combined_path)
		combined_path = Flatten()(combined_path)
		combined_path = Dropout(0.5)(combined_path) #2019.12.19 initially 0.3
		combined_path = Dense(512, activation='relu')(combined_path)

	combined_path = Dropout(0.5)(combined_path) #2019.12.19 initially 0.2
	combined_path = Dense(NUM_CLASSES, activation='softmax')(combined_path)

	if feature_size: #including epigenomic information
		model = Model(inputs=(sequence1_input, sequence2_input, epigenome1_input, epigenome2_input), outputs=combined_path)
	else:
		model = Model(inputs=(sequence1_input, sequence2_input), outputs=combined_path)
		
	if summary:
		print model.summary()

	return model

def model6_unifiedConvolution_original(feature_size=1, summary=True):
	fs_sequence = 40 # filter size
	padding_sequence = fs_sequence/2
	fs_epigenome = 10 # filter size
	padding_epigenome = fs_epigenome/2
	sequence_size = (SEQ_LENGTH, ONEHOT)
	if feature_size:
		epigenome_size = (EPIGENOME_LENGTH, feature_size)

	### Input setting
	sequence1_input	= Input(shape=sequence_size, name='sequence1')
	sequence2_input	= Input(shape=sequence_size, name='sequence2')
	sequence1_input_reshape = Reshape((SEQ_LENGTH, ONEHOT, 1))(sequence1_input)
	sequence2_input_reshape = Reshape((SEQ_LENGTH, ONEHOT, 1))(sequence2_input)
	sequence_path = concatenate([sequence1_input_reshape, sequence2_input_reshape], axis=-1)
	sequence_path = ZeroPadding2D(((padding_sequence, padding_sequence-1), (0,0)))(sequence_path)
	if feature_size:
		epigenome1_input	= Input(shape=epigenome_size, name='epigenome1')
		epigenome2_input	= Input(shape=epigenome_size, name='epigenome2')
		epigenome1_input_reshape = Reshape((EPIGENOME_LENGTH, feature_size, 1))(epigenome1_input)
		epigenome2_input_reshape = Reshape((EPIGENOME_LENGTH, feature_size, 1))(epigenome2_input)
		epigenome_path = concatenate([epigenome1_input_reshape, epigenome2_input_reshape], axis=-1)
		epigenome_path = ZeroPadding2D(((padding_epigenome, padding_epigenome-1), (0,0)))(epigenome_path)

	### SEQ PATH
	sequence_path = Conv2D(filters=512, kernel_size=(fs_sequence,4), padding='valid', activation='relu')(sequence_path)
	sequence_path = MaxPooling2D((50,1))(sequence_path)

	### EPIGENOME PATH 2019.10.03 every epigenomic information such as DNaase, CTCF, DNAmeth are combined into single path
	if feature_size:
		epigenome_path = Conv2D(filters=512, kernel_size=(fs_epigenome,feature_size), padding='valid', activation='relu')(epigenome_path)
		epigenome_path = MaxPooling2D((2,1))(epigenome_path)

	### COMBINED PATH
	if feature_size: #including epigenomic information
		combined_path = concatenate([sequence_path, epigenome_path], axis=2)
		combined_path = Conv2D(filters=1024, kernel_size=(10,2), padding='valid', activation='relu')(combined_path)
		combined_path = MaxPooling2D((2,1))(combined_path)
		combined_path = Flatten()(combined_path)
		combined_path = Dropout(0.3)(combined_path) #2019.12.19 initially 0.3
		combined_path = Dense(1024, activation='relu')(combined_path)
	else: #only sequence
		combined_path = sequence_path
		combined_path = Conv2D(filters=512, kernel_size=(10,1), padding='valid', activation='relu')(combined_path)
		combined_path = MaxPooling2D((2,1))(combined_path)
		combined_path = Flatten()(combined_path)
		#combined_path = Dropout(0.5)(combined_path) #2019.12.19 initially 0.3
		combined_path = Dropout(0.5)(combined_path) #2019.12.19 initially 0.3
		combined_path = Dense(512, activation='relu')(combined_path)

	#combined_path = Dropout(0.5)(combined_path) #2019.12.19 initially 0.2
	combined_path = Dropout(0.2)(combined_path) #2019.12.19 initially 0.2
	combined_path = Dense(NUM_CLASSES, activation='softmax')(combined_path)

	if feature_size: #including epigenomic information
		model = Model(inputs=(sequence1_input, sequence2_input, epigenome1_input, epigenome2_input), outputs=combined_path)
	else:
		model = Model(inputs=(sequence1_input, sequence2_input), outputs=combined_path)
		
	if summary:
		print model.summary()

	return model


def _model8_sharedLayer(feature_size=1):
	sequence_fs = 512 #1024
	sequence_fs2 = 512
	epigenome_fs = 512
	combined_fs = 1024 #2048

	fs_sequence = 40 # filter size
	padding_sequence = fs_sequence/2
	fs_epigenome = 10 # filter size
	padding_epigenome = fs_epigenome/2
	sequence_size = (SEQ_LENGTH, ONEHOT)
	if feature_size:
		epigenome_size = (EPIGENOME_LENGTH, feature_size)

	### Input setting
	sequence1_input	= Input(shape=sequence_size, name='sequence1')
	sequence2_input	= Input(shape=sequence_size, name='sequence2')
	sequence1_input_reshape = Reshape((SEQ_LENGTH, ONEHOT, 1))(sequence1_input)
	sequence2_input_reshape = Reshape((SEQ_LENGTH, ONEHOT, 1))(sequence2_input)
	sequence_path = concatenate([sequence1_input_reshape, sequence2_input_reshape], axis=-1)
	sequence_path = ZeroPadding2D(((padding_sequence, padding_sequence-1), (0,0)))(sequence_path)
	sequence_path = Conv2D(filters=sequence_fs, kernel_size=(fs_sequence,4), padding='valid', activation='relu', name='sequence_Conv2D_%d'%sequence_fs)(sequence_path)
	sequence_path = MaxPooling2D((50,1))(sequence_path)
	combined_path = sequence_path

	if feature_size:
		epigenome1_input	= Input(shape=epigenome_size, name='epigenome1')
		epigenome2_input	= Input(shape=epigenome_size, name='epigenome2')
		epigenome1_input_reshape = Reshape((EPIGENOME_LENGTH, feature_size, 1))(epigenome1_input)
		epigenome2_input_reshape = Reshape((EPIGENOME_LENGTH, feature_size, 1))(epigenome2_input)
		epigenome_path = concatenate([epigenome1_input_reshape, epigenome2_input_reshape], axis=-1)
		epigenome_path = ZeroPadding2D(((padding_epigenome, padding_epigenome-1), (0,0)))(epigenome_path)
		epigenome_path = Conv2D(filters=epigenome_fs, kernel_size=(fs_epigenome,feature_size), padding='valid', activation='relu', name='epigenome_Conv2D_%d'%epigenome_fs)(epigenome_path)
		epigenome_path = MaxPooling2D((2,1))(epigenome_path)
		combined_path = concatenate([sequence_path, epigenome_path], axis=2, name='combined_Concatenation')
		#combined_path = Conv2D(filters=combined_fs, kernel_size=(10,2), padding='valid', activation='relu', name='combined_Conv2D_%d'%combined_fs)(combined_path)
	else:
		combined_path = Conv2D(filters=sequence_fs2, kernel_size=(10,2), padding='valid', activation='relu', name='sequence_Conv2D_%d'%sequence_fs2)(combined_path)

	#combined_path = Dropout(0.5)(combined_path)
	#combined_path = MaxPooling2D((2,1))(combined_path)
	#combined_path = Flatten()(combined_path)

	if feature_size:
		return sequence1_input, sequence2_input, epigenome1_input, epigenome2_input, combined_path
	else:
		return sequence1_input, sequence2_input, combined_path

def _model8_separateLayer(previous_layer, feature_size=1, cellType=''):
	if feature_size: #including epigenomic information
		combined_path = Conv2D(filters=1024, kernel_size=(10,2), padding='valid', activation='relu', name='combined_Conv2D_separate')(previous_layer)
		combined_path = Dropout(0.5)(combined_path)
		combined_path = MaxPooling2D((2,1))(combined_path)
		combined_path = Flatten()(combined_path)
		#combined_path = Dense(1024, activation='relu', name='combinedDense')(previous_layer)
		combined_path = Dense(512, activation='relu', name='combinedDense1')(combined_path)
		combined_path = Dropout(0.3)(combined_path)
		combined_path = Dense(256, activation='relu', name='combinedDense2')(combined_path) # extra layer
	else: #only sequence
		combined_path = Dense(512, activation='relu', name='sequenceDense1')(previous_layer)
		combined_path = Dropout(0.5)(combined_path)
		combined_path = Dense(256, activation='relu', name='sequenceDense2')(combined_layer) # extra layer
	combined_path = Dropout(0.2)(combined_path)
	combined_path = Dense(NUM_CLASSES, activation='softmax', name='classificationLayer_%s'%cellType)(combined_path)

	return combined_path

def model8_unifiedConvolution_multiTasking(optimizer, input_cellType, feature_size=1, summary=True):
	cellType_list = ['E017','E116','E117','E119','E122','E123','E127']
	if input_cellType in cellType_list:
		cellType_list.remove(input_cellType)
	### shared layer declaration
	if feature_size:
		sequence1_input, sequence2_input, epigenome1_input, epigenome2_input, shared_layer = _model8_sharedLayer(feature_size=feature_size)	
	else:
		sequence1_input, sequence2_input, shared_layer = _model8_sharedLayer(feature_size=feature_size)	

	### specific layer define, per cell type
	model_dict = dict()
	for cellType in cellType_list:
		specific_classification_layer = _model8_separateLayer(shared_layer, feature_size=feature_size, cellType=cellType)
		if feature_size: #including epigenomic information
			model = Model(inputs=(sequence1_input, sequence2_input, epigenome1_input, epigenome2_input), outputs=specific_classification_layer)
		else:
			model = Model(inputs=(sequence1_input, sequence2_input), outputs=specific_classification_layer)
		### model compile
		model.compile(
			loss='binary_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy']
		)
		model_dict[cellType] = model	
		
		if summary and cellType == 'E017':
			print model.summary()
		else:
			print 'model summary HIDE for #', cellType

	return model_dict
