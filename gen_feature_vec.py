#!/usr/bin/python3
import numpy as np
import os.path as osp
import time
import _pickle as pickle
import sys


def gen_feature_vec(pattern_file, feature_vec_file_path, embedding):
	ev_diff_count = defaultdict(list)
	feature_vec1 = dict()
	progress=0

	with open(pattern_file) as f:
		patterns,pattern2patternid = [],{} # pattern index
		for line in f:
			row=line.rstrip('\n').split('\t')
			progress+=1
			if progress %1000000 ==0:
				print('progress = %s\n' % progress)
			if row[3] not in embedding or row[4] not in embedding :
				continue
			# find a new pattern
			if row[0] not in feature_vec1:
				pattern = row[0]+'\t'+row[1]+'\t'+row[2]
				row[0]=row[0].replace('_',' ')
				pattern_t = row[0].split()
				
				pattern_t = [element for i, element in enumerate(pattern_t) if i not in (row[1], row[2])]

				# The value of feature_vec1 is a numpy array
				pattern_embed=[embedding[word] for word in pattern_t if word in embedding]

				# if the pattern contains no valid words, skip the rest
				if not pattern_embed:
					continue
				# valid new patterns. Map pattern to id
				
				

				if not pattern in pattern2patternid:
					patterns.append(pattern)
					pattern2patternid[pattern] = len(pattern2patternid)
				patternid = pattern2patternid[pattern]
				# calculate the pattern embedding: mean of its word imbedding
				feature_vec1[patternid] =np.mean(pattern_embed, axis=0).tolist()
			# for new or old valid patterns, find the attribute embeddings: embedding_entity -embedding_value
			patternid = pattern2patternid[pattern]
			#if row[3] in embedding and row[4] in embedding:
			ev_diff_count[patternid].append(((embedding[row[3]] - embedding[row[4]]).tolist(), row[5]))
		
		# after get embeddings for all patterns and their extractions
		print('length of patterns: %s\n' % len(patterns))
		with open(feature_vec_file_path+'pattern2patternid'+ '.pickle', 'wb') as fp:
			pickle.dump(pattern2patternid, fp)
		with open(feature_vec_file_path+'patternid2pattern'+ '.pickle', 'wb') as fp:
			pickle.dump(patterns, fp)

		feature_vec2 = {pattern: np.mean([tpl[0] for tpl in ev_diff_count[pattern]], axis=0).tolist() for pattern in ev_diff_count}

		print('length of patterns (feature1): %s\n' % len(feature_vec1))
		with open(feature_vec_file_path+'feature1'+ '.pickle', 'wb') as fp:
			pickle.dump(feature_vec1, fp)
		print('length of patterns (feature2_mean): %s\n' % len(feature_vec2))
		with open(feature_vec_file_path+'feature2_mean'+ '.pickle', 'wb') as fp:
			pickle.dump(feature_vec2, fp)




if __name__ == '__main__':
	pattern_file = sys.argv[1] # pattern file
	feature_vec_file_path = sys.argv[3] # feature 
	embedding_file = sys.argv[2]
	print('reading in embedding file')
	if embedding_file[-3:]=='txt':
		embedding = pd.read_csv(embedding_file, delim_whitespace=True, header=None, quoting=csv.QUOTE_NONE)
		embedding_dict = dict()

		for row in embedding.values:
			embedding_dict[row[0]] = row[1:]

		with open(embedding_file[0:-4]+'.pickle', 'wb') as fp:
			pickle.dump(embedding_dict, fp)
	else:
		embedding_dict = pickle.load(open(embedding_file,'rb'))
	gen_feature_vec(pattern_file_path, feature_vec_file_path, embedding_dict)


