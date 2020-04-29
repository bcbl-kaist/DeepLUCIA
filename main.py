import os, sys, operator, random
import argparse
import numpy as np
from datetime import datetime

### sklearn kit import
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, History
from keras.callbacks import Callback
from keras.optimizers import Adam, Adagrad, RMSprop, Adadelta
from keras.metrics import categorical_accuracy

### inhouse import
import data_module
import model_framework

def progressPrint(input_str, num_margin=34, size_output=20):
	input_str = ' ' + input_str + ' '
	print ''
	print input_str.center(98,'#')

def str2bool(v):
	if isinstance(v, bool):
	 return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def train_LOOP(train_positive_loop, train_negative_loop, test_positive_loop, test_negative_loop, train_positive_sequence, train_negative_sequence, train_positive_epigenome, train_negative_epigenome, test_positive_sequence, test_negative_sequence, test_positive_epigenome, test_negative_epigenome, fold, positionSwitchAllowed=False):
	### setting seed as args.seed
	random.seed(args.seed)

	### SPLITTING DATASET X FOLD
	LEN_positive = (train_positive_sequence[0].shape[0]-train_positive_sequence[0].shape[0]/fold)
	LEN_negative = (train_negative_sequence[0].shape[0]-train_negative_sequence[0].shape[0]/fold)

	### sequence X fold validation split
	validation_positive_sequence = [train_positive_sequence[0][LEN_positive:],train_positive_sequence[1][LEN_positive:]]
	validation_negative_sequence = [train_negative_sequence[0][LEN_negative:],train_negative_sequence[1][LEN_negative:]]
	train_positive_sequence = [train_positive_sequence[0][:LEN_positive],train_positive_sequence[1][:LEN_positive]]
	train_negative_sequence = [train_negative_sequence[0][:LEN_negative],train_negative_sequence[1][:LEN_negative]]
	test_label = np.asarray([1]*test_positive_sequence[0].shape[0] + [0]*test_negative_sequence[0].shape[0])
	train_label = np.asarray([1]*train_positive_sequence[0].shape[0] + [0]*train_negative_sequence[0].shape[0])
	validation_label = np.asarray([1]*validation_positive_sequence[0].shape[0] + [0]*validation_negative_sequence[0].shape[0])
	if positionSwitchAllowed:
		print '# input1 and input2 position switching ALLOWED making switched dataset...'
		train_label = np.concatenate((train_label, train_label), axis=0)
		#validation_label = np.concatenate((validation_label, validation_label), axis=0)
	test_label = keras.utils.to_categorical(test_label, 2)
	train_label = keras.utils.to_categorical(train_label, 2)
	validation_label = keras.utils.to_categorical(validation_label, 2)

	print 'train Pos/Neg', np.sum(train_label, axis=0)
	print 'validation Pos/Neg', np.sum(validation_label, axis=0)
	
	### epigenome X fold validation split
	validation_positive_epigenome = [train_positive_epigenome[0][LEN_positive:],train_positive_epigenome[1][LEN_positive:]]
	validation_negative_epigenome = [train_negative_epigenome[0][LEN_negative:],train_negative_epigenome[1][LEN_negative:]]
	train_positive_epigenome = [train_positive_epigenome[0][:LEN_positive],train_positive_epigenome[1][:LEN_positive]]
	train_negative_epigenome = [train_negative_epigenome[0][:LEN_negative],train_negative_epigenome[1][:LEN_negative]]

	### MODEL fit
	test_sequence0 = np.concatenate((test_positive_sequence[0], test_negative_sequence[0]), axis=0)
	test_sequence1 = np.concatenate((test_positive_sequence[1], test_negative_sequence[1]), axis=0)
	test_epigenome0 = np.concatenate((test_positive_epigenome[0], test_negative_epigenome[0]), axis=0)
	test_epigenome1 = np.concatenate((test_positive_epigenome[1], test_negative_epigenome[1]), axis=0)
	feature_list.sort()
	if positionSwitchAllowed:
		train_sequence0 = np.concatenate((train_positive_sequence[0], train_negative_sequence[0], train_positive_sequence[1], train_negative_sequence[1]), axis=0)
		train_sequence1 = np.concatenate((train_positive_sequence[1], train_negative_sequence[1], train_positive_sequence[0], train_negative_sequence[0]), axis=0)
		train_epigenome0 = np.concatenate((train_positive_epigenome[0], train_negative_epigenome[0], train_positive_epigenome[1], train_negative_epigenome[1]), axis=0)
		train_epigenome1 = np.concatenate((train_positive_epigenome[1], train_negative_epigenome[1], train_positive_epigenome[0], train_negative_epigenome[0]), axis=0)
		### validation switching
		#validation_sequence0 = np.concatenate((validation_positive_sequence[0], validation_negative_sequence[0], validation_positive_sequence[1], validation_negative_sequence[1]), axis=0)
		#validation_sequence1 = np.concatenate((validation_positive_sequence[1], validation_negative_sequence[1], validation_positive_sequence[0], validation_negative_sequence[0]), axis=0)
		#validation_epigenome0 = np.concatenate((validation_positive_epigenome[0], validation_negative_epigenome[0], validation_positive_epigenome[1], validation_negative_epigenome[1]), axis=0)
		#validation_epigenome1 = np.concatenate((validation_positive_epigenome[1], validation_negative_epigenome[1], validation_positive_epigenome[0], validation_negative_epigenome[0]), axis=0)
		validation_sequence0 = np.concatenate((validation_positive_sequence[0], validation_negative_sequence[0]), axis=0)
		validation_sequence1 = np.concatenate((validation_positive_sequence[1], validation_negative_sequence[1]), axis=0)
		validation_epigenome0 = np.concatenate((validation_positive_epigenome[0], validation_negative_epigenome[0]), axis=0)
		validation_epigenome1 = np.concatenate((validation_positive_epigenome[1], validation_negative_epigenome[1]), axis=0)
	else:
		train_sequence0 = np.concatenate((train_positive_sequence[0], train_negative_sequence[0]), axis=0)
		train_sequence1 = np.concatenate((train_positive_sequence[1], train_negative_sequence[1]), axis=0)
		train_epigenome0 = np.concatenate((train_positive_epigenome[0], train_negative_epigenome[0]), axis=0)
		train_epigenome1 = np.concatenate((train_positive_epigenome[1], train_negative_epigenome[1]), axis=0)
		validation_sequence0 = np.concatenate((validation_positive_sequence[0], validation_negative_sequence[0]), axis=0)
		validation_sequence1 = np.concatenate((validation_positive_sequence[1], validation_negative_sequence[1]), axis=0)
		validation_epigenome0 = np.concatenate((validation_positive_epigenome[0], validation_negative_epigenome[0]), axis=0)
		validation_epigenome1 = np.concatenate((validation_positive_epigenome[1], validation_negative_epigenome[1]), axis=0)
		# testing reverse input
		validation_sequence0_r = np.concatenate((validation_positive_sequence[1], validation_negative_sequence[1]), axis=0)
		validation_sequence1_r = np.concatenate((validation_positive_sequence[0], validation_negative_sequence[0]), axis=0)
		validation_epigenome0_r = np.concatenate((validation_positive_epigenome[1], validation_negative_epigenome[1]), axis=0)
		validation_epigenome1_r = np.concatenate((validation_positive_epigenome[0], validation_negative_epigenome[0]), axis=0)

	print train_sequence0.shape, validation_sequence0.shape, train_label.shape, validation_label.shape

	print '\n# fitting using...', 'FEATURE_LIST=', feature_list
	progressPrint('Keras Learning Start')
	filepath='modelOUT' + '/model-%s_%s-bestModel.h5'%(time_tag, args.out)
	modelHistory = History()
	if args.savemodel:
		CALLBACKS=[modelHistory]
	else:
		CALLBACKS=[]	

	min_val_loss = np.inf
	for i in range(EPOCH):
		model.fit(
			x=[
				train_sequence0, train_sequence1,
				train_epigenome0, train_epigenome1,
			],
			y=train_label,
			batch_size=BATCH,
			epochs=1,
			verbose=VERBOSE,
			validation_data=[
				[
					validation_sequence0, validation_sequence1,
					validation_epigenome0, validation_epigenome1,
				], validation_label
			],
			shuffle=True,
			callbacks=CALLBACKS
		)

		if min_val_loss > modelHistory.history['val_loss'][0] and args.savemodel:
			print 'new val_loss=%5.4f is better than min_val_loss=%5.4f'%(modelHistory.history['val_loss'][0], min_val_loss)
			min_val_loss = modelHistory.history['val_loss'][0]
			model.save(filepath)
			y_test = model.predict(
				[
					test_sequence0, test_sequence1,
					test_epigenome0, test_epigenome1
				], verbose=0)
			print '#EPOCH%d test on %s AUC='%(i+1, args.testset), roc_auc_score(test_label, y_test), 'new val_loss=%5.4f is better than min_val_loss=%5.4f'%(modelHistory.history['val_loss'][0], min_val_loss)
		  print '#Classification report:'
		  print(classification_report(test_label.argmax(axis=-1), y_test.argmax(axis=-1)))
      sys.stdout.flush()

def getData_loop(cellType, feature_list, seed, positivePair_file, negativePair_file, data_limit, capping, testset, balanced=True):
	### DATA loading and final PREP
	loader = data_module.data_prep(
		ctype=cellType,
		feature_list=feature_list,
		seed=seed,
		neg_seed=seed,
		positivePair_file=positivePair_file,
		negativePair_file=negativePair_file,
		data_limit=data_limit,
		pCap=capping,
		balanced=balanced,
		chr_test=testset
	)	
	### data shape:	
	### 	sequence: input1_sequence
	### 	epigenome: input1_epigenome
	if len(feature_list):
		train_positive_loop, train_negative_loop, test_positive_loop, test_negative_loop, train_positive_sequence, train_negative_sequence, train_positive_epigenome, train_negative_epigenome, test_positive_sequence, test_negative_sequence, test_positive_epigenome, test_negative_epigenome = loader.loop()
	else:
		train_positive_loop, train_negative_loop, test_positive_loop, test_negative_loop, train_positive_sequence, train_negative_sequence, test_positive_sequence, test_negative_sequence = loader.loop()

	if len(feature_list):
		return train_positive_loop, train_negative_loop, test_positive_loop, test_negative_loop, train_positive_sequence, train_negative_sequence, train_positive_epigenome, train_negative_epigenome, test_positive_sequence, test_negative_sequence, test_positive_epigenome, test_negative_epigenome
	else:
		return train_positive_loop, train_negative_loop, test_positive_loop, test_negative_loop, train_positive_sequence, train_negative_sequence, test_positive_sequence, test_negative_sequence

if __name__ == '__main__':
	### various setting
	parser = argparse.ArgumentParser() 
	parser.add_argument('--features', type=str, default='', help='list of features to be used separated by , ; chose it from [H4K20me1,H2AFZ,H3K9ac,H3K27ac,H3K4me1,H3K4me2,H3K4me3,H3K9me3,H3K79me2,H3K27me3,H3K36me3,DNase,CTCF]')
	parser.add_argument('--cellType', type=str, default='E116', help='chose cell type to be trained: E017, E116, E117, E119, E122, E123, E127')
	parser.add_argument('--out', type=str, default='OUT', help='output extention args.out')
	parser.add_argument('--model', type=str, default='separateConvolution', help='select model framework to be trained')
	parser.add_argument('--optimizer', type=str, default='adam', help='select optimizer: adam, adadelta, adagrad, rmsprop')
	parser.add_argument('--epoch', type=int, default=10, help='SET epoch number for model training')
	parser.add_argument('--lr', type=float, default=0.001, help='SET Learning Rate for model training')
	parser.add_argument('--seed', type=int, default=777, help='SET seed for data shuffling')
	parser.add_argument('--saveDB', type=str2bool,  default=False, help='If declared, shuffled train, validation data would ve saved')
	parser.add_argument('--balanced', type=str2bool,  default=False, help='If declared, shuffled train, validation data would ve saved')
	parser.add_argument('--type_positive', type=str, default='loop', help='pair type to be used for POSITIVE defualt=loop')
	parser.add_argument('--type_negative', type=str, default='singleAnchor', help='pair type to be used for NEGATIVE defaulst=singleAnchor can be chosed from [singleAnchor, doubleAnchor, None')
	parser.add_argument('--pair_path', type=str, default='/home/cosmos/research_3D_genome/20191104_cellTypeModel_GenerateNegative/pairLabel_cellType/', help='path setting where bin pair files are saved')
	parser.add_argument('--fold', type=int, default=5, help='X fold cross validation model training, DEFAULT=5')
	parser.add_argument('--verbose', type=int, default=2, help='verbose setting for training, [1:detailed, 2:simplified, 0:no output], DEFAULT=2')
	parser.add_argument('--batch', type=int, default=100, help='set batch size for training, DEFAULT=100')
	parser.add_argument('--data_limit', type=int, default=-1, help='set limit of data usage of positive and negative set')
	parser.add_argument('--capping', type=float, default=-1, help='set capping of epigenoem, if -1 no capping will be applied, DEFAULT=-1')
	parser.add_argument('--savemodel', type=str2bool, default=True, help='save model per epoch, DEFAULT=True')
	parser.add_argument('--testset', type=str, default='chr1', help='select chromosome number to be used for external data set, DEFAULT=chr1')
	parser.add_argument('--position_switch', type=str2bool, default=False, help='input1 and input2 will be switched to consider various inputs and data size will be doubled, DEFAULT=False')
	args = parser.parse_args()

	#### data size ####
	NUM_CLASSES = 2
	SEQ_LENGTH = 5000
	EPIGENOME_LENGTH = 200
	ONEHOT = 4
	#### model training parameters
	EPOCH = args.epoch
	VERBOSE = args.verbose
	BATCH = args.batch

	#### get whole cell types
	f = open('../cellType.txt', 'r')
	cellType_list = [x.strip() for x in f.readlines()]
	f.close()

	### check epigenomes to be used for prediction
	whole_feature_list = [
		'H4K20me1',
		'H2AFZ',
		'H3K9ac',
		'H3K27ac',
		'H3K4me1',
		'H3K4me2',
		'H3K4me3',
		'H3K9me3',
		'H3K79me2',
		'H3K27me3',
		'H3K36me3',
		'DNase',
		#'DNAmeth',
		#'CTCF'
	]

	feature_list = args.features.split(',')
	if len(feature_list) == 1 and feature_list[0] == 'all':
		feature_list = list(whole_feature_list)
	elif len(feature_list) == 1 and feature_list[0] == '':
		feature_list = []
	elif len(feature_list) == 1 and feature_list[0] == 'sequence':
		feature_list = []
	else:
		for single in feature_list:
			if single not in whole_feature_list:
				print 'Wrong feature name', single
				sys.exit()

	progressPrint('Preparation Start')
	print '\n# Selected features to be used for TRAINING'
	if len(feature_list):
		for x in feature_list:
			print '>', x
	else:
		print 'Only using sequence'

	progressPrint('Model Construction')
	### Selection of model type
	#if args.model == 'separateConvolution':
	#	model = model_framework.model1_separateConvolution(feature_size=len(feature_list))
	#elif args.model == 'combinedConvolution':
	#	model = model_framework.model2_combinedConvolution(feature_size=len(feature_list))
	#elif args.model == 'combinedConvolutionLSTM':
	#	model = model_framework.model3_combinedConvolutionLSTM(feature_size=len(feature_list))
	#elif args.model == 'combinedConvolutionMore':
	#	model = model_framework.model4_combinedConvolutionMore(feature_size=len(feature_list))
	#elif args.model == 'unifiedConvolution_single':
	#	model = model_framework.model7_unifiedConvolution(feature_size=len(feature_list))
	#elif args.model == 'unifiedConvolution_multitask':
	#	model = model_framework.model8_unifiedConvolution_multiTasking(optimizer=Adam(0.0001), input_cellType='', feature_size=len(feature_list))
	if args.model == 'unifiedConvolution':
		model = model_framework.model6_unifiedConvolution(feature_size=len(feature_list))
	elif args.model == 'unifiedConvolution_original':
		model = model_framework.model6_unifiedConvolution_original(feature_size=len(feature_list))

	progressPrint('Data Loading')
	### check if multiple cellTypes are given as input
	args.cellType = args.cellType.split(',')
	positivePair_file=[args.pair_path + x + '_%s.txt'%args.type_positive for x in args.cellType]
	### check if negativePair_file is None, loop, singleAnchor, doubleAnchor
	if args.type_negative == 'none':
		negativePair_file=args.pair_path + 'None.txt'
	elif args.type_negative == 'loop':
		negativePair_file = list()
		for cellType in cellType_list:
			if cellType in args.cellType:
				continue
			negativePair_file.append(args.pair_path + cellType + '_loop.txt')
	elif args.type_negative == 'singleAnchor':
		negativePair_file = list()
		for cellType in args.cellType:
			negativePair_file.append(args.pair_path + cellType + '_singleAnchor.txt')
	elif args.type_negative == 'doubleAnchor': # 2019.12.17 too small dataset ~1,000
		negativePair_file = list()
		for cellType in args.cellType:
			negativePair_file.append(args.pair_path + cellType + '_doubleAnchor.txt')
	#else:
	#	negativePair_file=args.pair_path + args.cellType + '_%s.txt'%args.type_negative
	
	print '\n# input positive pair file: %s \n# input negative pair file: %s'%(positivePair_file, negativePair_file)

	train_positive_loop, train_negative_loop, test_positive_loop, test_negative_loop, train_positive_sequence, train_negative_sequence, train_positive_epigenome, train_negative_epigenome, test_positive_sequence, test_negative_sequence, test_positive_epigenome, test_negative_epigenome = getData_loop(
		args.cellType,
		feature_list,
		args.seed,
		positivePair_file,
		negativePair_file,
		args.data_limit,
		args.capping,
		args.testset,
		balanced=args.balanced,
	)

	### MODEL TRAINING
	progressPrint('Model Training')

	### Model parameter
	if args.optimizer == 'adam': # later lr: learning rate should be changable
		print '\n# Optimizer= adam, with LearningRate= %f'%args.lr
		optimizer = Adam(args.lr)
	elif args.optimizer == 'adagrad':
		print '\n# Optimizer= adagrad, with LearningRate= %f'%args.lr
		optimizer = Adagrad(args.lr)
	elif args.optimizer == 'rmsprop':
		print '\n# Optimizer= rmsprop, with LearningRate= %f'%args.lr
		optimizer = RMSprop(args.lr)	
	elif args.optimizer == 'adadelta':
		print '\n# Optimizer= adadelta, with LearningRate= %f'%args.lr
		optimizer = Adadelta(args.lr)
		
	model.compile(
		loss='binary_crossentropy',
		optimizer=optimizer,
		metrics=['accuracy']
	)

	train_LOOP(train_positive_loop, train_negative_loop, test_positive_loop, test_negative_loop, train_positive_sequence, train_negative_sequence, train_positive_epigenome, train_negative_epigenome, test_positive_sequence, test_negative_sequence, test_positive_epigenome, test_negative_epigenome, args.fold, positionSwitchAllowed=args.position_switch)

