#!/usr/bin/env python

import gzip

def load_chrominfo(chrominfo_filename):
	chrom_to_size = {}
	with open(chrominfo_filename,"rt") as chrominfo_file:
		for rawline in chrominfo_file:
			fields = rawline.strip().split()
			chrom = fields[0]
			size = int(fields[1])
			chrom_to_size[chrom] = size

	return chrom_to_size

def is_chrom_of(locus_index_pair,index_range):
	return min(index_range) <= min(locus_index_pair) & max(locus_index_pair) <= max(index_range)


def is_in_desired_range_set(sample_locus_index_pair,index_range_set):
	return any([ is_chrom_of(sample_locus_index_pair[1],index_range) for index_range in index_range_set ])
	

def get_locus_chrom(locus_index,chrom_to_locus_index_range):
	return [chrom for chrom,locus_index_range in chrom_to_locus_index_range.items() if locus_index_range[0] <= locus_index <= locus_index_range[1]][0]
	

def is_intra_chrom(sample_locus_index_pair,chrom_to_locus_index_range):
	locus_index_one,locus_index_two = sample_locus_index_pair[1]
	chrom_one = get_locus_chrom(locus_index_one,chrom_to_locus_index_range)
	chrom_two = get_locus_chrom(locus_index_two,chrom_to_locus_index_range)
	return chrom_one == chrom_two


def is_anchor_bearing(sample_locus_index_pair,anchor_locus_index_set):
	return not(set(sample_locus_index_pair[1]) & anchor_locus_index_set)


def get_sample_index(sample_list):
	sample_to_sample_index = {}
	for sample_index,sample in enumerate(sample_list):
		sample_to_sample_index[sample] = sample_index
	return sample_to_sample_index

