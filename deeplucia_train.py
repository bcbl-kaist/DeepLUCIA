#!/usr/bin/env python

import sys
import json
from types import SimpleNamespace


def main():
	if len(sys.argv) < 2:
		print("Usage : " + sys.argv[0] + " [config_json_filename]")
		sys.exit()
	else:
		config_json_filename = sys.argv[1]
		deeplucia_train(config_json_filename)

def deeplucia_train(config_json_filename):

	import os
	import functools
	import itertools

	from pathlib import Path

	from functional import seq
	from functional import pseq # seq with parallelism 

	from deeplucia_toolkit import prep_matrix
	from deeplucia_toolkit import prep_label
	from deeplucia_toolkit import make_dataset
	from deeplucia_toolkit import make_model
	from deeplucia_toolkit import misc

	import tensorflow as tf

	from tensorflow import keras
	from tensorflow.keras import layers
	from tensorflow.keras import regularizers
	from tensorflow.keras.models import Model
	from tensorflow.keras.callbacks import ModelCheckpoint
	from tensorflow.keras.callbacks import EarlyStopping
	from tensorflow.keras.callbacks import CSVLogger
	from tensorflow.keras.callbacks import Callback


	if not os.path.isdir("modelOUT/"):
		os.makedirs("modelOUT/")
	if not os.path.isdir("history/"):
		os.makedirs("history/")

	# open config file 
	with open(config_json_filename) as config_json_file:
		config_dict = json.load(config_json_file)
		config = SimpleNamespace(**config_dict)

	# obvious parameter setting 
	local_path_base = "/content/" if os.path.exists("/content") else "/home/schloss/ver_20191030/"
	chrominfo_filename = "/home/bcbl_commons/CDL/static/ChromInfo/ChromInfo_hg19.txt"
	chrom_set = set(["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22","chrX"])
	
	validate_chrom_set =  set(config.val_chrom_list)
	test_chrom_set = set(config.test_chrom_list)
	train_chrom_set = chrom_set - test_chrom_set - validate_chrom_set
	
	n2p_ratio = config.n2p_ratio
	#num_pos = config.num_pos 
	num_pos = max(1,int(config.num_pos/config.n2p_ratio))

	# set filepath
	loop_label_txtgz_filename = Path(local_path_base) / "Label" / "isHC.loop_list.txt.gz"
	anchor_label_txtgz_filename = Path(local_path_base) / "Label" / "isHC.anchor_list.txt.gz"

	seq_numpy_filename = Path(local_path_base) / "GenomicFeature" / "isHC" / "isHC.seq_onehot.npy"

	sample_list = config.sample_id_list
	mark_list = ["DNase","H2AFZ","H3K27ac","H3K27me3","H3K36me3","H3K4me1","H3K4me2","H3K4me3","H3K79me2","H3K9ac","H3K9me3","H4K20me1"]
	sample_mark_to_epi_numpy_filename = {}
	for epi_numpy_filename in Path(local_path_base).glob("EpigenomicFeature/*/*.npy"):
		sample_mark = tuple(epi_numpy_filename.name.split(".")[:2])
		sample_mark_to_epi_numpy_filename[sample_mark] = epi_numpy_filename

	sample_to_sample_index = misc.get_sample_index(sample_list)

	# load feature array
	seq_array = prep_matrix.load_seq_array(seq_numpy_filename)
	multisample_multimark_epi_array = prep_matrix.load_multisample_multimark_epi_array(sample_list,mark_list,sample_mark_to_epi_numpy_filename,cap_crit=0.95)
	print("load feature array => DONE")

	# load loop label
	chrom_to_size = misc.load_chrominfo(chrominfo_filename)

	pos_sample_locus_index_pair_set = prep_label.load_loop_label(sample_list,loop_label_txtgz_filename)
	anchor_locus_index_set = prep_label.get_anchor_locus(sample_list,pos_sample_locus_index_pair_set)
	chrom_to_locus_index_range= prep_label.get_chrom_range(anchor_label_txtgz_filename,chrom_to_size)

	index_max = max(itertools.chain(*list(chrom_to_locus_index_range.values())))

	test_locus_index_range_set = {chrom_to_locus_index_range[chrom] for chrom in test_chrom_set}
	train_locus_index_range_set = {chrom_to_locus_index_range[chrom] for chrom in train_chrom_set}
	validate_locus_index_range_set = {chrom_to_locus_index_range[chrom] for chrom in validate_chrom_set}
	print("load loop label => DONE")

	# curryize basic functions
	is_intra_chrom = functools.partial(misc.is_intra_chrom,chrom_to_locus_index_range=chrom_to_locus_index_range)
	is_anchor_bearing = functools.partial(misc.is_anchor_bearing,anchor_locus_index_set=anchor_locus_index_set)
	permute_same_distance = functools.partial(prep_label.permute_same_distance,index_max = index_max)
	gen_neg_sample_locus_index_pair = functools.partial(prep_label.gen_neg_sample_locus_index_pair,sample_list=sample_list,index_max=index_max)

	is_in_test_range_set = functools.partial(misc.is_in_desired_range_set,index_range_set=test_locus_index_range_set)
	is_in_train_range_set = functools.partial(misc.is_in_desired_range_set,index_range_set=train_locus_index_range_set)
	is_in_validate_range_set = functools.partial(misc.is_in_desired_range_set,index_range_set=validate_locus_index_range_set)

	def permute_pos_sample_locus_index_pair(pos_sample_locus_index_pair,n2p_ratio,is_in_range_set):
		neg_sample_locus_index_pair_list = (
			seq(gen_neg_sample_locus_index_pair(pos_sample_locus_index_pair))
			.filter(is_in_range_set)
			.filter(is_intra_chrom)
			.filter_not(is_anchor_bearing)
			.take(n2p_ratio)
			.to_list())
		return neg_sample_locus_index_pair_list

	permute_pos_sample_locus_index_pair_test = functools.partial(permute_pos_sample_locus_index_pair,n2p_ratio = n2p_ratio,is_in_range_set = is_in_test_range_set)
	permute_pos_sample_locus_index_pair_train = functools.partial(permute_pos_sample_locus_index_pair,n2p_ratio = n2p_ratio,is_in_range_set = is_in_train_range_set)
	permute_pos_sample_locus_index_pair_validate = functools.partial(permute_pos_sample_locus_index_pair,n2p_ratio = n2p_ratio,is_in_range_set = is_in_validate_range_set)
	print("curryize basic functions => DONE")

	# split train/validate label
	train_pos_sample_locus_index_pair_list = (pseq(pos_sample_locus_index_pair_set).filter(is_in_train_range_set).to_list())
	train_neg_sample_locus_index_pair_list = (pseq(train_pos_sample_locus_index_pair_list).flat_map(permute_pos_sample_locus_index_pair_train).to_list())

	validate_pos_sample_locus_index_pair_list = (pseq(pos_sample_locus_index_pair_set).filter(is_in_validate_range_set).to_list())
	validate_neg_sample_locus_index_pair_list = (pseq(validate_pos_sample_locus_index_pair_list).flat_map(permute_pos_sample_locus_index_pair_validate).to_list())
	print("split test/train/validate label => DONE")

	# merge pos/neg label
	train_sample_locus_index_pair_list_gen = prep_label.gen_sample_locus_index_pair_list(train_pos_sample_locus_index_pair_list,train_neg_sample_locus_index_pair_list,num_pos,n2p_ratio)
	validate_sample_locus_index_pair_list_gen = prep_label.gen_sample_locus_index_pair_list(validate_pos_sample_locus_index_pair_list,validate_neg_sample_locus_index_pair_list,num_pos,n2p_ratio)
	print("merge pos/neg label => DONE")

	# make dataset
	train_dataset_gen = make_dataset.gen_seq_epi_dataset(train_sample_locus_index_pair_list_gen,pos_sample_locus_index_pair_set,seq_array,multisample_multimark_epi_array,sample_to_sample_index)
	validate_dataset_gen = make_dataset.gen_seq_epi_dataset(validate_sample_locus_index_pair_list_gen,pos_sample_locus_index_pair_set,seq_array,multisample_multimark_epi_array,sample_to_sample_index)
	print("make dataset => DONE")

	# prepare model
	model =make_model.seq_epi_20210105_v1_model()
	print("load model => DONE")

	# keras parameter setting 

	adam_optimizer = keras.optimizers.Adam(lr=0.0001)
	modelCheckpoint = ModelCheckpoint(filepath='modelOUT/model_seq_epi_20210105_v1_' + config.log_id + '_{epoch:05d}.h5', verbose=1)
	csvLogger = CSVLogger("history/model_seq_epi_20210105_v1_" + config.log_id + ".csv")

	weight_for_0 = ( n2p_ratio + 1 )/(2*n2p_ratio)
	weight_for_1 = ( n2p_ratio + 1 )/2
	class_weight = {0: weight_for_0, 1: weight_for_1}

	print("keras parameter setting => DONE")


	# model compile and fitting
	model.compile(
					loss='binary_crossentropy',
					metrics=["accuracy", "Precision" , "Recall" , "TruePositives" , "FalsePositives" , "FalseNegatives" , "AUC"],
					optimizer=adam_optimizer)

	model_history = model.fit(train_dataset_gen,
					steps_per_epoch = 400,
					epochs=1000,
					validation_data = validate_dataset_gen,
					validation_steps= 100,
					class_weight=class_weight,
					callbacks=[modelCheckpoint,csvLogger]
					)


if __name__ == "__main__":
	main()

