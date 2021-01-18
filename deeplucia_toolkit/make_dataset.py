#!/usr/bin/env python

import itertools 

import numpy
from tensorflow import keras

def take_seq_array(total_seq_array,sample_locus_index_pair_list):

	locus_index_one_list = [sample_locus_index_pair[1][0] for sample_locus_index_pair in sample_locus_index_pair_list]
	locus_index_two_list = [sample_locus_index_pair[1][1] for sample_locus_index_pair in sample_locus_index_pair_list]
	selected_seq_array_one = numpy.take(total_seq_array,locus_index_one_list,0).astype(dtype=numpy.float32)*0.25
	selected_seq_array_two = numpy.take(total_seq_array,locus_index_two_list,0).astype(dtype=numpy.float32)*0.25

	return selected_seq_array_one,selected_seq_array_two

def take_epi_array(total_multisample_multimark_epi_array,sample_locus_index_pair_list,sample_to_sample_index):
	sample_index_list = [sample_to_sample_index[sample_locus_index_pair[0]] for sample_locus_index_pair in sample_locus_index_pair_list]
	locus_index_one_list = [sample_locus_index_pair[1][0] for sample_locus_index_pair in sample_locus_index_pair_list]
	locus_index_two_list = [sample_locus_index_pair[1][1] for sample_locus_index_pair in sample_locus_index_pair_list]

	selected_epi_array_one = total_multisample_multimark_epi_array[tuple(numpy.moveaxis(list(zip(sample_index_list,locus_index_one_list)),-1,0))]
	selected_epi_array_two = total_multisample_multimark_epi_array[tuple(numpy.moveaxis(list(zip(sample_index_list,locus_index_two_list)),-1,0))]

	return selected_epi_array_one,selected_epi_array_two


#######################################################

def extract_seq_epi_dataset(sample_locus_index_pair_list,pos_sample_locus_index_pair_set,total_seq_array,total_multisample_multimark_epi_array,sample_to_sample_index):
	numpy.random.shuffle(sample_locus_index_pair_list)
	label = list(map(lambda sample_locus_index_pair : 1 if sample_locus_index_pair in pos_sample_locus_index_pair_set else 0,sample_locus_index_pair_list))
	label = numpy.asarray(label)
	#label = keras.utils.to_categorical(label, 2) # for original

	seq_feature_one,seq_feature_two = take_seq_array(total_seq_array,sample_locus_index_pair_list)
	epi_feature_one,epi_feature_two = take_epi_array(total_multisample_multimark_epi_array,sample_locus_index_pair_list,sample_to_sample_index)

	return ((seq_feature_one,seq_feature_two,epi_feature_one,epi_feature_two),label)


def gen_seq_epi_dataset(sample_locus_index_pair_list_gen,pos_sample_locus_index_pair_set,total_seq_array,total_multisample_multimark_epi_array,sample_to_sample_index):
	for sample_locus_index_pair_list in sample_locus_index_pair_list_gen:
		yield extract_seq_epi_dataset(sample_locus_index_pair_list,pos_sample_locus_index_pair_set,total_seq_array,total_multisample_multimark_epi_array,sample_to_sample_index)


def extract_seq_epi_dataset_unshuffled(sample_locus_index_pair_list,pos_sample_locus_index_pair_set,total_seq_array,total_multisample_multimark_epi_array,sample_to_sample_index):
	label = list(map(lambda sample_locus_index_pair : 1 if sample_locus_index_pair in pos_sample_locus_index_pair_set else 0,sample_locus_index_pair_list))
	label = numpy.asarray(label)
	seq_feature_one,seq_feature_two = take_seq_array(total_seq_array,sample_locus_index_pair_list)
	epi_feature_one,epi_feature_two = take_epi_array(total_multisample_multimark_epi_array,sample_locus_index_pair_list,sample_to_sample_index)
	return ((seq_feature_one,seq_feature_two,epi_feature_one,epi_feature_two),label)


def extract_seq_epi_dataset_unlabeled(sample_locus_index_pair_list,total_seq_array,total_multisample_multimark_epi_array,sample_to_sample_index):
	seq_feature_one,seq_feature_two = take_seq_array(total_seq_array,sample_locus_index_pair_list)
	epi_feature_one,epi_feature_two = take_epi_array(total_multisample_multimark_epi_array,sample_locus_index_pair_list,sample_to_sample_index)
	return (seq_feature_one,seq_feature_two,epi_feature_one,epi_feature_two)




def gen_sample_locus_index_pair_sublist(sample_locus_index_pair_list):
	for _,chunk in itertools.groupby(enumerate(sample_locus_index_pair_list), lambda x: x[0]// 5 ):
		yield [indexed_sample_locus_index_pair[1] for indexed_sample_locus_index_pair in chunk]

def gen_seq_epi_subdataset(sample_locus_index_pair_sublist_gen,pos_sample_locus_index_pair_set,total_seq_array,total_multisample_multimark_epi_array,sample_to_sample_index):
	for i,sample_locus_index_pair_sublist in enumerate(sample_locus_index_pair_sublist_gen):
		#print(str(i) + " : " + str(sample_locus_index_pair_sublist))
		feature,label = extract_seq_epi_dataset(sample_locus_index_pair_sublist,pos_sample_locus_index_pair_set,total_seq_array,total_multisample_multimark_epi_array,sample_to_sample_index)
		print(len(feature))
		yield feature


def extract_seq_epi_chunked_dataset(sample_locus_index_pair_list,pos_sample_locus_index_pair_set,total_seq_array,total_multisample_multimark_epi_array,sample_to_sample_index):
	label = list(map(lambda sample_locus_index_pair : 1 if sample_locus_index_pair in pos_sample_locus_index_pair_set else 0,sample_locus_index_pair_list))
	label = numpy.asarray(label)

	sample_locus_index_pair_sublist_gen = gen_sample_locus_index_pair_sublist(sample_locus_index_pair_list)
	seq_epi_subdataset_gen = gen_seq_epi_subdataset(sample_locus_index_pair_sublist_gen,pos_sample_locus_index_pair_set,total_seq_array,total_multisample_multimark_epi_array,sample_to_sample_index)
	return (seq_epi_subdataset_gen,label)


"""
def extract_seq_epi_dataset2(sample_locus_index_pair_list,pos_sample_locus_index_pair_set,total_seq_array,total_multisample_multimark_epi_array,sample_to_sample_index):
	numpy.random.shuffle(sample_locus_index_pair_list)
	label = list(map(lambda sample_locus_index_pair : 1 if sample_locus_index_pair in pos_sample_locus_index_pair_set else 0,sample_locus_index_pair_list))
	label = numpy.asarray(label)
	label = keras.utils.to_categorical(label, 2) # for original

	seq_feature_one,seq_feature_two = take_seq_array(total_seq_array,sample_locus_index_pair_list)
	epi_feature_one,epi_feature_two = take_epi_array(total_multisample_multimark_epi_array,sample_locus_index_pair_list,sample_to_sample_index)

	return ((seq_feature_one,seq_feature_two,epi_feature_one,epi_feature_two),label)
"""
