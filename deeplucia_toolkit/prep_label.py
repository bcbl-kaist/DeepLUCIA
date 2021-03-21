#!/usr/bin/env python

import gzip
import itertools
import random

def load_loop_label(sample_list,loop_label_txtgz_filename):
	from pathlib import Path
	import json
	import hashlib
	import pickle 

	sample_list_hash = hashlib.md5(json.dumps(sample_list).encode("utf-8")).hexdigest()
	multisample_loop_pickle_filename = Path(Path.cwd() / "InputMatrix" / (sample_list_hash + ".loop.pickle"))
	if multisample_loop_pickle_filename.exists():
		print(multisample_loop_pickle_filename)
		with open(multisample_loop_pickle_filename,"rb") as multisample_loop_pickle_file:
			pos_sample_locus_index_pair_set = pickle.load(multisample_loop_pickle_file)
	else:
		pos_sample_locus_index_pair_set = set()

		given_sample_set = set(sample_list)
		with gzip.open(loop_label_txtgz_filename,"rt") as loop_label_txtgz_file:
			for rawline in itertools.islice(loop_label_txtgz_file,1,None):
				fields = rawline.strip().split()
				if fields[4] != "None":
					locus_index_one = int(fields[2])-1
					locus_index_two = int(fields[3])-1
					locus_index_pair = (locus_index_one,locus_index_two)
					sample_set = set(fields[4].split(",")) & given_sample_set
					for sample in sample_set:
						sample_locus_index_pair = sample,locus_index_pair
						pos_sample_locus_index_pair_set.add(sample_locus_index_pair)

		with open(multisample_loop_pickle_filename,"wb") as multisample_loop_pickle_file:
			pickle.dump(pos_sample_locus_index_pair_set,multisample_loop_pickle_file,protocol=pickle.HIGHEST_PROTOCOL)

	return pos_sample_locus_index_pair_set


def get_anchor_locus(sample_list,pos_sample_locus_index_pair_set):
	anchor_locus_index_set = set()
	given_sample_set = set(sample_list)
	for sample_locus_index_pair in pos_sample_locus_index_pair_set:
		sample,locus_index_pair = sample_locus_index_pair
		if sample in given_sample_set:
			locus_index_one,locus_index_two = locus_index_pair
			anchor_locus_index_set.add(locus_index_one)
			anchor_locus_index_set.add(locus_index_two)

	return anchor_locus_index_set


def get_chrom_range(anchor_label_txtgz_filename,chrom_to_size):
	chrom_to_locus_index_range = {chrom:(1000000,0) for chrom in chrom_to_size.keys()}
	with gzip.open(anchor_label_txtgz_filename,"rt") as anchor_label_txtgz_file:
		for rawline in itertools.islice(anchor_label_txtgz_file,1,None):
			fields = rawline.split()
			locus_index = int(fields[0])-1
			chrom = fields[1].split(":")[0]
			chrom_to_locus_index_range[chrom] = (min(chrom_to_locus_index_range[chrom][0],locus_index),max(locus_index,chrom_to_locus_index_range[chrom][1]))
	return 	chrom_to_locus_index_range


def permute_same_distance(locus_index_pair,index_max):
	loop_length = locus_index_pair[1]-locus_index_pair[0]
	permuted_locus_index_one = random.randint(0,index_max-loop_length)
	permuted_locus_index_two = permuted_locus_index_one + loop_length 
	permuted_locus_index_pair = (permuted_locus_index_one , permuted_locus_index_two)
	return permuted_locus_index_pair


def gen_neg_sample_locus_index_pair(pos_sample_locus_index_pair,sample_list,index_max):
	pos_locus_index_pair = pos_sample_locus_index_pair[1]
	while True: 
		neg_sample = random.choice(sample_list)
		neg_locus_index_pair = permute_same_distance(pos_locus_index_pair,index_max)
		neg_sample_locus_index_pair = (neg_sample,neg_locus_index_pair)
		yield neg_sample_locus_index_pair


def gen_sample_locus_index_pair_list(pos_sample_locus_index_pair_list,neg_sample_locus_index_pair_list,num_pos,n2p_ratio):
	while True:
		num_neg = int(num_pos * n2p_ratio)
		selected_pos_sample_locus_index_pair_list = random.sample(pos_sample_locus_index_pair_list,num_pos)
		selected_neg_sample_locus_index_pair_list = random.sample(neg_sample_locus_index_pair_list,num_neg)
		yield(list(selected_pos_sample_locus_index_pair_list) + list(selected_neg_sample_locus_index_pair_list))

