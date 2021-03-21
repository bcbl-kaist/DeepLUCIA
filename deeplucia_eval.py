#!/usr/bin/env python

import sys
import json
from types import SimpleNamespace



def main():
	if len(sys.argv) < 3:
		print("Usage : " + sys.argv[0] + " [keras_model_filename] [config_json_filename]")
		sys.exit()
	else:
		keras_model_filename = sys.argv[1]
		config_json_filename = sys.argv[2]
		deeplucia_eval(keras_model_filename,config_json_filename)

def deeplucia_eval(keras_model_filename,config_json_filename):

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
	from tensorflow.keras.models import Model

	import numpy

	from sklearn.metrics import precision_score
	from sklearn.metrics import recall_score
	from sklearn.metrics import f1_score
	

	from sklearn.metrics import average_precision_score
	from sklearn.metrics import roc_auc_score
	from sklearn.metrics import confusion_matrix
	from sklearn.metrics import matthews_corrcoef


	# open config file 
	with open(config_json_filename) as config_json_file:
		config_dict = json.load(config_json_file)
		config = SimpleNamespace(**config_dict)

	# obvious parameter setting 
	local_path_base = Path (Path.cwd() / "Features")
	chrominfo_filename = local_path_base / "ChromInfo_hg19.txt"
	chrom_set = set(["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22","chrX"])
	
	test_chrom_set = set(config.test_chrom_list)
	
	n2p_ratio = config.n2p_ratio
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
	print("load loop label => DONE")

	# curryize basic functions
	is_intra_chrom = functools.partial(misc.is_intra_chrom,chrom_to_locus_index_range=chrom_to_locus_index_range)
	is_anchor_bearing = functools.partial(misc.is_anchor_bearing,anchor_locus_index_set=anchor_locus_index_set)
	permute_same_distance = functools.partial(prep_label.permute_same_distance,index_max = index_max)
	gen_neg_sample_locus_index_pair = functools.partial(prep_label.gen_neg_sample_locus_index_pair,sample_list=sample_list,index_max=index_max)

	is_in_test_range_set = functools.partial(misc.is_in_desired_range_set,index_range_set=test_locus_index_range_set)
	
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
	print("curryize basic functions => DONE")

	# split train/validate label
	test_pos_sample_locus_index_pair_list = (pseq(pos_sample_locus_index_pair_set).filter(is_in_test_range_set).to_list())
	test_neg_sample_locus_index_pair_list = (pseq(test_pos_sample_locus_index_pair_list).flat_map(permute_pos_sample_locus_index_pair_test).to_list())

	print("split test/train/validate label => DONE")

	# merge pos/neg label
	test_sample_locus_index_pair_list = test_pos_sample_locus_index_pair_list + test_neg_sample_locus_index_pair_list

	print("merge pos/neg label => DONE")


	# prepare model
	model = keras.models.load_model(keras_model_filename)
	print("load model => DONE")

	prob_pred_list = []
	label_pred_list = []
	label_true_list = []

	chunk_size = 10000

	for _,chunk in itertools.groupby(enumerate(test_sample_locus_index_pair_list), lambda x: x[0]// chunk_size ):
		test_sample_locus_index_pair_sublist = [ indexed_sample_locus_index_pair[1] for indexed_sample_locus_index_pair in chunk ]
		feature_test,label_true = make_dataset.extract_seq_epi_dataset_unshuffled(test_sample_locus_index_pair_sublist,pos_sample_locus_index_pair_set,seq_array,multisample_multimark_epi_array,sample_to_sample_index)
		output = model.predict(feature_test)
		prob_pred = numpy.squeeze(output,axis=1)
		label_pred = list(map(lambda prob: int(round(prob)), prob_pred))

		prob_pred_list.append(prob_pred)
		label_pred_list.append(label_pred)
		label_true_list.append(label_true)

	prob_pred = list(itertools.chain(*prob_pred_list))
	label_pred = list(itertools.chain(*label_pred_list))
	label_true = list(itertools.chain(*label_true_list))


	f1 = f1_score (label_true ,label_pred)
	mcc = matthews_corrcoef(label_true ,label_pred)
	au_ro_curve = roc_auc_score(label_true ,prob_pred)
	au_pr_curve = average_precision_score(label_true ,prob_pred)

	model_evaluation_filename = "eval_model/" + keras_model_filename.split("/")[-1] + "." + config.log_id + ".xls"

	with open(model_evaluation_filename,"wt") as model_evaluation_file:

		model_evaluation_file.write(keras_model_filename.split("/")[-1][:-20] + "\t" + config.log_id + "\tAUROC\t" + str(au_ro_curve)+"\n")
		model_evaluation_file.write(keras_model_filename.split("/")[-1][:-20] + "\t" + config.log_id + "\tAUPRC\t" + str(au_pr_curve)+"\n")
		model_evaluation_file.write(keras_model_filename.split("/")[-1][:-20] + "\t" + config.log_id + "\tMCC\t" + str(mcc)+"\n")
		model_evaluation_file.write(keras_model_filename.split("/")[-1][:-20] + "\t" + config.log_id + "\tF1\t" + str(f1)+"\n")


if __name__ == "__main__":
	main()
















