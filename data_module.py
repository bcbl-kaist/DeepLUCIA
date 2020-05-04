import os, sys, gzip, random, time
import numpy as np

def merge_dict(input_dict1, input_dict2):
	keys = list(set(input_dict1.keys() + input_dict2.keys()))
	keys.sort()
	new_dict = {key:[] for key in keys}
	for key in keys:
		if key in input_dict1:
			new_dict[key].extend(input_dict1[key])
		if key in input_dict2:
			new_dict[key].extend(input_dict2[key])
	return new_dict

def print_dict(input_dict, type='len'):
	for key, value in input_dict.iteritems():
		if type=='len':
			print key, len(value)

def sequence_load(file_name):
	print 'Sequence file list:', [file_name]
	print 'loading sequence file...', file_name
	tmp_data = np.load(file_name)
	return tmp_data

def epigenomeNORM_pCapping_paired(input_epigenome1, input_epigenome2, cutoff=0.95):
	mid_point = input_epigenome1.shape[0]
	epigenome_data = np.concatenate((input_epigenome1, input_epigenome2), axis=0)/1000
	tmp = epigenome_data.flatten()
	tmp.sort()
	LEN = tmp.shape[0] # positive and negative sets are balanced.
	cap = tmp[int(LEN-LEN*(1.0-cutoff))]
	tmp = epigenome_data
	tmp[tmp>cap] = cap
	tmp /= cap
	epigenome_data = tmp
	return epigenome_data[:mid_point], epigenome_data[mid_point:]

def epigenomeNORM_pCapping(input_epigenome, cutoff=0.95):
	epigenome_data = input_epigenome/1000
	tmp = epigenome_data.flatten()
	tmp.sort()
	LEN = tmp.shape[0] # positive and negative sets are balanced.
	cap = tmp[int(LEN-LEN*(1.0-cutoff))]
	print '### CAP', cap, tmp[0], tmp[-1]
	tmp = epigenome_data
	tmp[tmp>cap] = cap
	tmp /= cap
	epigenome_data = tmp
	return epigenome_data
	
def epigenome_load(file_list, selected_epigenome_info='all', pCap=0.95):
	if selected_epigenome_info == 'all':
		pass
	else:
		file_list = [x for x in file_list if x in selected_epigenome_info]
	file_list.sort()
	epigenome_order = list(set([x.split('.')[-2] for x in file_list]))
	epigenome_order.sort()
	print 'Epigenome file list:', file_list

	data_container = []
	for i, epiType in enumerate(epigenome_order):
		file_name = [x for x in file_list if epiType in x][0]
		print 'loading epigenome file...', file_name
		tmp_data = np.load(file_name)
		tmp_data = tmp_data.astype(dtype=np.float32)
		if len(data_container) == 0:
			data_container = np.zeros((tmp_data.shape[0], tmp_data.shape[1], len(epigenome_order)), dtype=np.float32)
		if pCap == -1:
			pass
		else:
			print '### CAPPING', i, epiType
			tmp_data = epigenomeNORM_pCapping(tmp_data, cutoff=pCap)
		data_container[:,:,i] = tmp_data

	return data_container

def get_file_recursive(path):
	whole_file_list = []
	for (dir_path, dirnames, filenames) in os.walk(path):
		whole_file_list.extend([dir_path + '/' + x for x in filenames])
	return whole_file_list


############################################## CLASS ##########################################################
class data_prep:
	def __init__(self, ctype='E116', feature_list=[], neg_seed=111, seed=777, get_pairIndex=False, negativePair_file='/home/cosmos/research_3D_genome/20191104_cellTypeModel_GenerateNegative/pairLabel_file/E116_singleAnchor.txt', positivePair_file='/home/cosmos/research_3D_genome/20191104_cellTypeModel_GenerateNegative/pairLabel_cellType/E116_loop.txt', balanced=True, data_limit=-1, pCap=0.95, chr_test='chr1'):
		self.sequence_file = '/home/bcbl_commons/CDL/ver_20191030/GenomicFeature/isHC/isHC.seq_onehot.npy'	
		self.epigenomic_path = '/home/bcbl_commons/CDL/ver_20191030/EpigenomicFeature/'
		self.epigenome_files = [x for x in get_file_recursive(self.epigenomic_path) if x.split('/')[-1].split('.')[0] in ctype and x.split('.')[-2] in feature_list]

		self.chr_index_file = 'chr_index.txt'
		self.pCap = pCap
		self.seed = seed
		self.cellType = ctype
		self.SEQ_LENGTH = 5000	
		self.EPIGENOME_LENGTH = 200
		self.EPIGENOME_SIZE = len(feature_list)
		self.feature_list = feature_list
		self.sequence_data_container = ''
		self.epignome_data_container = ''
		self.data_limit=data_limit

		### get pair list from given Pair_file, self redundancy is checked using set 
		self.positive_pair, self.positive_pair_dict, self.positive_pair_set = self._get_pair(positivePair_file)
		self.negative_pair, self.negative_pair_dict, self.negative_pair_set = self._get_pair(negativePair_file)
	
		### redundancy eliminated negative pair
		self.negative_pair = self. _redundancy_check_negative(self.negative_pair)
		print '\n# data info cellType=%s'%self.cellType, 'balanced dataset=', balanced
		self.positive_pair_dict = self._redundancy_check_dict(self.positive_pair_dict)
		self.negative_pair_dict = self._redundancy_check_dict(self.negative_pair_dict, check_with_positive=True)
	
		print '# positive dict'
		print_dict(self.positive_pair_dict)
		print 'sum', np.sum([len(x) for x in self.positive_pair_dict.values()])
		print '# negative dict'
		print_dict(self.negative_pair_dict)
		print 'sum', np.sum([len(x) for x in self.negative_pair_dict.values()])
		#== after this part, pair_list of positive and negative are compeletely non-overlapping

		### shuffling of negative pair using time seed and make balanced set
		#random.seed(round(time.time())) ## for consistancy random shuffle is blocked for temparary
		#random.seed(neg_seed) ## for consistancy random shuffle is blocked for temparary
		#random.shuffle(self.negative_pair)
		if balanced:
			self.negative_pair = self.negative_pair[:len(self.positive_pair)]
			self.negative_pair_dict = self.trim_pair_dict(self.negative_pair_dict, self.negative_pair)
		else:
			self.negative_pair = self.negative_pair[:len(self.positive_pair)*10]
			self.negative_pair_dict = self.trim_pair_dict(self.negative_pair_dict, self.negative_pair)
		#== after this part, negative pair list is set to be same size as positive set = balanced dataset

		### reading chromosom index file and min max index of chromosoms are collected for indexing
		f = open(self.chr_index_file, 'r')
		tmp_stream = [x.strip() for x in f.readlines()]
		f.close()
		chr_id_minmax = dict()
		for line in tmp_stream:
			chr_id = line.split()[0]
			id_min = int(line.split('=')[-2].split()[0])
			id_max = int(line.split('=')[-1].strip())
			chr_id_minmax[chr_id] = (id_min-1, id_max-1)

		self.positive_pair.sort()
		self.negative_pair.sort()
		print '###', chr_test
		print chr_id_minmax[chr_test]
		self.test_positive_pair, self.train_positive_pair = self._split_pair_by_index(self.positive_pair, index_threshold=chr_id_minmax[chr_test])
		self.test_negative_pair, self.train_negative_pair = self._split_pair_by_index(self.negative_pair, index_threshold=chr_id_minmax[chr_test])

		### print summary
		print 'TRUE pair:', len(self.test_positive_pair), len(self.train_positive_pair)
		print 'FALSE pair:', len(self.test_negative_pair), len(self.train_negative_pair)

	def _shuffle_index(self, input_array):
		random.seed(self.seed)
		indices = range(input_array.shape[0])
		random.shuffle(indices)
		print self.seed, input_array.shape, indices[:10]
		input_array = input_array[indices]
		return input_array

	def _redundancy_check_dict(self, tmp_pair_dict, check_with_positive=False): # if multiple cell types were found
		tmp_checked = {}
		tmp_used = set([])
		for key, value in tmp_pair_dict.iteritems():
			tmp_value = set(map(tuple, value))
			tmp_unique = tmp_value.difference(tmp_used)
			if check_with_positive:
				tmp_unique = tmp_unique.difference(self.positive_pair_set)
			tmp_used = tmp_used.union(tmp_unique)
			tmp_unique = map(list, tmp_unique)
			tmp_checked[key] = tmp_unique
		
		return tmp_checked

	def _redundancy_check_negative(self, tmp_pair_list):
		tmp_set = set(map(tuple, tmp_pair_list))
		tmp_set = tmp_set.difference(self.positive_pair_set)
		return map(list, tmp_set)
		
	def trim_pair_dict(self, pair_dict, pair_list):
		pair_list = set(map(tuple, pair_list))
		for key, value in pair_dict.iteritems():
			tmp_set = set(map(tuple, value))
			tmp_set = tmp_set.intersection(pair_list)
			pair_dict[key] = map(list, tmp_set)
		return pair_dict

	def _split_pair_by_index(self, pair_list, index_threshold=(0,49842)):
		test_pair = []
		train_pair = []
		test_index = [i for i, x in enumerate(pair_list) if index_threshold[0] <= x[0] <= index_threshold[1] and index_threshold[0] <= x[1] <= index_threshold[1]]
		train_index = [i for i, x in enumerate(pair_list) if i not in test_index]	
		return np.asarray(pair_list)[test_index].tolist(), np.asarray(pair_list)[train_index].tolist()

	def _get_pair(self, pair_file, shuffle=False):
		pair = list()
		pair_dict = dict()
		if isinstance(pair_file, list):
			for single_file in pair_file:
				single_cellType = single_file.split('/')[-1].split('_')[0]
				if single_cellType == 'None.txt':
					single_cellType = self.cellType[0]
				tmp_pair_list = list()
				with open(single_file, 'r') as f:		
					for line in f.readlines():
						tmp_pair_list.append(map(int, line.split()[:2]))
				if shuffle:
					random.seed(round(time.time()))
					idx = range(len(tmp_pair_list))
					random.shuffle(idx)
					tmp_pair_list = np.asarray(tmp_pair_list)[idx].tolist()
				pair.extend(tmp_pair_list)
				pair_dict[single_cellType] = tmp_pair_list	
		elif isinstance(pair_file, str):
			single_cellType = pair_file.split('/')[-1].split('_')[0]
			if single_cellType == 'None.txt':
				single_cellType = self.cellType[0]
			tmp_pair_list = list()
			with open(pair_file, 'r') as f:		
				for line in f.readlines():
					tmp_pair_list.append(map(int, line.split()[:2]))
				if shuffle:
					random.seed(round(time.time()))
					idx = range(len(tmp_pair_list))
					random.shuffle(idx)
					tmp_pair_list = np.asarray(tmp_pair_list)[idx].tolist()
				pair.extend(tmp_pair_list)
				pair_dict[single_cellType] = tmp_pair_list	
		pair_set = set(map(tuple, pair))
		pair.sort()
		
		return pair, pair_dict, pair_set

	def _anchor_sequence(self, anchor_list):
		if len(self.sequence_data_container) == 0:
			print '\n# loading and preparing Sequence data'
			self.sequence_data_container = sequence_load(self.sequence_file)
		seq = np.zeros((len(anchor_list), self.SEQ_LENGTH, 4), dtype=np.float32)

		### DATA SPLIT and SEQ into float32
		for i, anchor in enumerate(anchor_list):
			tmp_anchor_seq = self.sequence_data_container[anchor].astype(dtype=np.float32)/4.0
			seq[i] = tmp_anchor_seq[0]

		if self.data_limit == -1:
			return seq
		elif self.data_limit > 0:
			return seq[:self.data_limit]

	def _anchor_epigenome(self, anchor_list):
		if len(self.epignome_data_container) == 0:
			print '\n# loading and preparing Epigenome data'
			self.epignome_data_container = epigenome_load(self.epigenome_files, pCap=self.pCap)

		epigenome = np.zeros((len(anchor_list), self.EPIGENOME_LENGTH, self.EPIGENOME_SIZE), dtype=np.float32)
		### DATA SPLIT and SEQ into float32
		for i, anchor in enumerate(anchor_list):
			tmp_anchor_epigenome = self.epignome_data_container[anchor]
			epigenome[i] = tmp_anchor_epigenome[0]

		if self.data_limit == -1:
			return epigenome
		elif self.data_limit > 0:
			return epigenome[:self.data_limit]
			
	def _whole_sequence(self, pair_list):
		if len(self.sequence_data_container) == 0:
			print '\n# loading and preparing Sequence data'
			self.sequence_data_container = sequence_load(self.sequence_file)
		seq_1 = np.zeros((len(pair_list), self.SEQ_LENGTH, 4), dtype=np.float32)
		seq_2 = np.zeros((len(pair_list), self.SEQ_LENGTH, 4), dtype=np.float32)

		### DATA SPLIT and SEQ into float32
		for i, pair in enumerate(pair_list):
			tmp_pair_seq = self.sequence_data_container[pair].astype(dtype=np.float32)/4.0
			seq_1[i] = tmp_pair_seq[0]
			seq_2[i] = tmp_pair_seq[1]

		if self.data_limit == -1:
			return seq_1, seq_2
		elif self.data_limit > 0:
			return seq_1[:self.data_limit], seq_2[:self.data_limit]

	def _whole_epigenome(self, pair_list):
		if len(self.epignome_data_container) == 0:
			print '\n# loading and preparing Epigenome data'
			self.epignome_data_container = epigenome_load(self.epigenome_files, pCap=self.pCap)

		epigenome_1 = np.zeros((len(pair_list), self.EPIGENOME_LENGTH, self.EPIGENOME_SIZE), dtype=np.float32)
		epigenome_2 = np.zeros((len(pair_list), self.EPIGENOME_LENGTH, self.EPIGENOME_SIZE), dtype=np.float32)
		### DATA SPLIT and SEQ into float32
		for i, pair in enumerate(pair_list):
			tmp_pair_epigenome = self.epignome_data_container[pair]
			epigenome_1[i] = tmp_pair_epigenome[0]
			epigenome_2[i] = tmp_pair_epigenome[1]

		if self.data_limit == -1:
			return epigenome_1, epigenome_2
		elif self.data_limit > 0:
			return epigenome_1[:self.data_limit], epigenome_2[:self.data_limit]
			

	def loop(self, shuffle=True):
		test_positive_loop = self.test_positive_pair
		test_negative_loop = self.test_negative_pair
		test_positive_sequence_data = self._whole_sequence(self.test_positive_pair)
		test_negative_sequence_data = self._whole_sequence(self.test_negative_pair)
		train_positive_loop = self.train_positive_pair
		train_negative_loop = self.train_negative_pair
		train_positive_sequence_data = self._whole_sequence(self.train_positive_pair)
		train_negative_sequence_data = self._whole_sequence(self.train_negative_pair)
		if len(self.feature_list):
			test_positive_epigenome_data = self._whole_epigenome(self.test_positive_pair)
			test_negative_epigenome_data = self._whole_epigenome(self.test_negative_pair)
			train_positive_epigenome_data = self._whole_epigenome(self.train_positive_pair)
			train_negative_epigenome_data = self._whole_epigenome(self.train_negative_pair)

		if len(self.feature_list):
			return train_positive_loop, train_negative_loop, test_positive_loop, test_negative_loop, train_positive_sequence_data, train_negative_sequence_data, train_positive_epigenome_data, train_negative_epigenome_data, test_positive_sequence_data, test_negative_sequence_data, test_positive_epigenome_data, test_negative_epigenome_data
		else:
			return train_positive_loop, train_negative_loop, test_positive_loop, test_negative_loop, train_positive_sequence_data, train_negative_sequence_data, test_positive_sequence_data, test_negative_sequence_data
			
		

	### 2020.03.17 note: FUNCTION anchor created for anchor loading train sets (test=chr1), non redundancy added
	def anchor(self, shuffle=True):
		### get non redundant anchor list for test and train set
		test_positive_anchor = np.asarray(self.test_positive_pair).flatten()
		test_negative_anchor = np.asarray(self.test_negative_pair).flatten()

		print '### Cheking redundancy'
		print test_positive_anchor.shape
		test_positive_anchor = list(set(test_positive_anchor.tolist()))
		print len(test_positive_anchor)
		test_negative_anchor = list(set(test_negative_anchor.tolist()))[:len(test_positive_anchor)*10]

		train_positive_anchor = np.asarray(self.train_positive_pair).flatten()
		train_negative_anchor = np.asarray(self.train_negative_pair).flatten()
		print train_positive_anchor.shape
		train_positive_anchor = list(set(train_positive_anchor.tolist()))
		print len(train_positive_anchor)
		train_negative_anchor = list(set(train_negative_anchor.tolist()))[:len(train_positive_anchor)*10]

		### create label dataset matches anchor list
		#test_label = np.asarray([1]*len(test_positive_anchor) + [0]*len(test_negative_anchor))
		#train_label = np.asarray([1]*len(train_positive_anchor) + [0]*len(train_negative_anchor))

		### load sequence data of anchor
		print '### LOADING SEQUENCE'
		test_positive_sequence_data = self._anchor_sequence(test_positive_anchor)
		test_negative_sequence_data = self._anchor_sequence(test_negative_anchor)
		train_positive_sequence_data = self._anchor_sequence(train_positive_anchor)
		train_negative_sequence_data = self._anchor_sequence(train_negative_anchor)

		print '### LOADING EPIGENOME'
		if len(self.feature_list):
			test_positive_epigenome_data = self._anchor_epigenome(test_positive_anchor)
			test_negative_epigenome_data = self._anchor_epigenome(test_negative_anchor)
			train_positive_epigenome_data = self._anchor_epigenome(train_positive_anchor)
			train_negative_epigenome_data = self._anchor_epigenome(train_negative_anchor)

			train_positive_anchor = np.asarray(train_positive_anchor)
			train_negative_anchor = np.asarray(train_negative_anchor)
			test_positive_anchor = np.asarray(test_positive_anchor)
			test_negative_anchor = np.asarray(test_negative_anchor)

			#print '### TESTING'
			#print train_sequence_data.shape, train_epigenome_data.shape
			#print test_sequence_data.shape, test_epigenome_data.shape

		if shuffle:
			print '\n#### shuffling data'
			train_positive_anchor = self._shuffle_index(train_positive_anchor)
			train_negative_anchor = self._shuffle_index(train_negative_anchor)
			test_positive_anchor = self._shuffle_index(test_positive_anchor)
			test_negative_anchor = self._shuffle_index(test_negative_anchor)
			test_positive_sequence_data = self._shuffle_index(test_positive_sequence_data)
			test_negative_sequence_data = self._shuffle_index(test_negative_sequence_data)
			train_positive_sequence_data = self._shuffle_index(train_positive_sequence_data)
			train_negative_sequence_data = self._shuffle_index(train_negative_sequence_data)
			if len(self.feature_list):
				test_positive_epigenome_data = self._shuffle_index(test_positive_epigenome_data)
				test_negative_epigenome_data = self._shuffle_index(test_negative_epigenome_data)
				train_positive_epigenome_data = self._shuffle_index(train_positive_epigenome_data)
				train_negative_epigenome_data = self._shuffle_index(train_negative_epigenome_data)

		print '\n#### DATA loading complete'
		print 'anchor train positive/negative=', train_positive_anchor.shape, train_negative_anchor.shape
		print 'anchor test positive/negative=', test_positive_anchor.shape, test_negative_anchor.shape
		print 'sequence_data train positive/negative=', train_positive_sequence_data.shape, train_negative_sequence_data.shape
		print 'sequence_data test positive/negative=', test_positive_sequence_data.shape, test_negative_sequence_data.shape
		if len(self.feature_list):
			print 'epigenome_data train positive/negative=', train_positive_epigenome_data.shape, train_negative_epigenome_data.shape
			print 'epigenome_data test positive/negative=', test_positive_epigenome_data.shape, test_negative_epigenome_data.shape

		if len(self.feature_list):
			return train_positive_anchor, train_negative_anchor, test_positive_anchor, test_negative_anchor, train_positive_sequence_data, train_negative_sequence_data, train_positive_epigenome_data, train_negative_epigenome_data, test_positive_sequence_data, test_negative_sequence_data, test_positive_epigenome_data, test_negative_epigenome_data
		else:
			return train_positive_anchor, train_negative_anchor, test_positive_anchor, test_negative_anchor, train_positive_sequence_data, train_negative_sequence_data, test_positive_sequence_data, test_negative_sequence_data

		

if __name__ == '__main__':
	loader = data_prep(sys.argv[1])
