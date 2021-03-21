#!/usr/bin/env python

import numpy 

def load_seq_array(seq_numpy_filename):
	return numpy.load(seq_numpy_filename,mmap_mode="r")

def load_epi_array(epi_numpy_filename):
	return numpy.load(epi_numpy_filename,mmap_mode="r")

def load_capped_epi_array(epi_numpy_filename , cap_crit = 0.95):
	epi_array = load_epi_array(epi_numpy_filename)
	cap_val = int(numpy.quantile(epi_array,cap_crit))
	epi_array = numpy.clip(epi_array,None,cap_val)
	return epi_array

def load_multimark_epi_array(mark_list,mark_to_epi_numpy_filename,cap_crit = 0.95):
	epi_array_list = []
	for mark in mark_list:
		epi_numpy_filename = mark_to_epi_numpy_filename[mark]
		epi_array = load_capped_epi_array(epi_numpy_filename , cap_crit)
		epi_array_list.append(epi_array)
	multimark_epi_array = numpy.stack(epi_array_list,axis=2)
	return multimark_epi_array

def load_multisample_multimark_epi_array(sample_list,mark_list,sample_mark_to_epi_numpy_filename,cap_crit=0.95):
	from pathlib import Path
	import json
	import hashlib

	sample_list_hash = hashlib.md5(json.dumps(sample_list).encode("utf-8")).hexdigest()
	mark_list_hash = hashlib.md5(json.dumps(mark_list).encode("utf-8")).hexdigest()
	multisample_multimark_epi_array_filename = Path(Path.cwd() / "InputMatrix" / (sample_list_hash + "." + mark_list_hash + ".npy"))
	print(multisample_multimark_epi_array_filename)
	if multisample_multimark_epi_array_filename.exists():
		multisample_multimark_epi_array = numpy.load(multisample_multimark_epi_array_filename,mmap_mode="r")
	else:
		multimark_epi_array_list = []
		sample_mark_to_epi_numpy_filename
		for sample in sample_list:
			mark_to_epi_numpy_filename = {sample_mark[1]:epi_numpy_filename for sample_mark,epi_numpy_filename in sample_mark_to_epi_numpy_filename.items() if sample_mark[0] == sample}
			multimark_epi_array = load_multimark_epi_array(mark_list,mark_to_epi_numpy_filename,cap_crit)
			multimark_epi_array_list.append(multimark_epi_array)
		multisample_multimark_epi_array = numpy.stack(multimark_epi_array_list,axis=0)
		multisample_multimark_epi_array = multisample_multimark_epi_array.astype(dtype=numpy.float32) *0.00005

		numpy.save(multisample_multimark_epi_array_filename,multisample_multimark_epi_array)

	return multisample_multimark_epi_array
