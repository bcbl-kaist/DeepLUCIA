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
		sample_id = sys.argv[2]
		scan_chrom = sys.argv[3]

		deeplucia_fullscan(keras_model_filename,sample_id,scan_chrom)

def deeplucia_fullscan(keras_model_filename,sample_id,scan_chrom):

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

	from sklearn.metrics import average_precision_score
	from sklearn.metrics import roc_auc_score
	from sklearn.metrics import confusion_matrix
	from sklearn.metrics import matthews_corrcoef

	# obvious parameter setting 
	local_path_base = Path (Path.cwd() / "Features")
	chrominfo_filename = local_path_base / "ChromInfo_hg19.txt"
	chrom_set = set(["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22","chrX"])
	scan_chrom_set = set([scan_chrom])

	# set filepath
	loop_label_txtgz_filename = Path(local_path_base) / "Label" / "isHC.loop_list.txt.gz"
	anchor_label_txtgz_filename = Path(local_path_base) / "Label" / "isHC.anchor_list.txt.gz"

	seq_numpy_filename = Path(local_path_base) / "GenomicFeature" / "isHC" / "isHC.seq_onehot.npy"

	sample_list = [sample_id]
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
	chrom_to_locus_index_range= prep_label.get_chrom_range(anchor_label_txtgz_filename,chrom_to_size)
	index_max = max(itertools.chain(*list(chrom_to_locus_index_range.values())))

	scan_locus_index_range_set = {chrom_to_locus_index_range[chrom] for chrom in scan_chrom_set}
	print("load loop label => DONE")

	# curryize basic functions
	is_intra_chrom = functools.partial(misc.is_intra_chrom,chrom_to_locus_index_range=chrom_to_locus_index_range)
	print("curryize basic functions => DONE")

	# prepare candiates
	index_scan_interval = chrom_to_locus_index_range[scan_chrom]
	scan_sample_locus_index_pair_list = ( (sample_id,locus_index_pair) for locus_index_pair in itertools.combinations(range(*tuple(index_scan_interval)),2) if 4<locus_index_pair[1]-locus_index_pair[0]<400 )
	scan_sample_locus_index_pair_list = [sample_locus_index_pair for sample_locus_index_pair in scan_sample_locus_index_pair_list if is_intra_chrom(sample_locus_index_pair)]
	print("prepare candiates => DONE")
	
	# prepare model
	model = keras.models.load_model(keras_model_filename)
	print("load model => DONE")
	
	loop_anno_list = []
	prob_pred_list = []
	
	chunk_size = 200

	pred_result_csv_filename = "predicted/" + sample_id + "." + scan_chrom + ".csv"
	with open(pred_result_csv_filename,"w",buffering=1) as pred_result_csv_file:
		pred_result_csv_file.write(",".join(["sample","anchor_one","anchor_two","prob_pred"])+"\n")

		for _,chunk in itertools.groupby(enumerate(scan_sample_locus_index_pair_list), lambda x: x[0]// chunk_size ):
			scan_sample_locus_index_pair_sublist = [ indexed_sample_locus_index_pair[1] for indexed_sample_locus_index_pair in chunk ]
			feature_scan = make_dataset.extract_seq_epi_dataset_unlabeled(scan_sample_locus_index_pair_sublist,seq_array,multisample_multimark_epi_array,sample_to_sample_index)
			output = model.predict(feature_scan)
			prob_pred = numpy.squeeze(output,axis=1)
			label_pred = list(map(lambda prob: int(round(prob)), prob_pred))

			for item in zip(scan_sample_locus_index_pair_sublist , prob_pred):
				pred_result_csv_file.write(",".join(map(str,[item[0][0],item[0][1][0],item[0][1][1],item[1]])) + "\n")


if __name__ == "__main__":
	main()
















