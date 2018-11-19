#!/usr/bin/python3
import tool
import os.path as osp
from io import open
import _pickle as pickle
import sys
import json
from sklearn.preprocessing import normalize
if __name__ == "__main__":

# 1. preprocessing
# 1.1 word2vec
# 1.2 NER
# 1.3 pattern mining
# 2. run truepie
# 2.1 positive patterns: precision>threshold
# 2.2 negative patterns: precision<threshold and min(dist(pattern embedding,positive patterns))>threshold
# 3. classification
# 3.1 run knn 
# 3.2 candidates patterns are those that at least contain one entity from positive pattern set
# 3.3 merge candidate patterns


	f = sys.argv[1]
	feature_path = sys.argv[2]
	output_IT = sys.argv[3]
	output_Step = sys.argv[4]
	output_Test = sys.argv[5]
	
	iterations=2
	
	task='leader' #task name
	e="$LOCATION" # entity type holder
	v="$PERSON" # value type holder
	gold_patterns = set()
	# to add user specified seed patterns:
	# gold_patterns.add('')
	c1=0.6 # penalty for infrequent patterns. [0,1). Higher means stronger penalty
	theta =0.8 # threshold for positive patterns. 
	theta2= 1-theta # threshold for negative patterns
	beta = 20 #penalty to overwrite soft constraint
	m_tau=2 #b. penalty to reduce therandomness in extractions
	constr='' #if empty, then learn from data
	# constr can be user specified. For example
	# constr='1 soft-1 soft'
	#constr='1 hard-50 soft'
	
	#train parameters
	theta3=0.9 #threshold for the maximal similarity between any positive pattens to a negative pattern. i.e., if a negative pattern has higher than 0.9 cosine similarity to any positive pattern, it is removed from negative pattern set
	nn=15 # parameter for Knn
	gold_threshold=0.51 # if P(y=1|x)>gold_threshold, then y=1; otherwise, y=-1
	
	
	
	#read in pattern file
	retrieve_tuples,patternSummary, patternsAll,pattern2patternidAll, entities,entity2entityid, values,value2valueid =tool.readMetaPAD(f)
	print('task: ' + task)
	
	# automatically construct seed patterns. optional
	most_freq = tool.find_most_freq(patternSummary,task) #most frequent pattern that contains the key word
	task_token= len(task.split(' '))
	gold_patterns={e+' ' +task + ' ' +v+'\t0\t'+str(task_token+1),}
	if most_freq:
		gold_patterns.add(most_freq)
	# to add user specified pattern:
	# gold_patterns.add('')


	print(gold_patterns)
	
	patternSummary=[[pattern2patternidAll[i],j] for i, j in patternSummary]
	patternSummary=dict(patternSummary)
	# run TruePIE: construct training
	E_list, positive_pattern, negative_pattern, constraint = tool.TruePIE (output_IT, task, retrieve_tuples, patternSummary, gold_patterns, patternsAll, pattern2patternidAll, c1, theta,theta2, beta, option,constraint=constr, minimal_tau=m_tau)

	# map back and write intermediate result

	seed_tpl=tool.writeResult(entities, values, E_list,path=osp.join(output_IT,task+'_tuple_result.txt'))#[entityid, valueid, weight]
	all_tple = tool.positiveP2T(positive_pattern, pattern2patternidAll, retrieve_tuples)
	print('read positive patterns')
	positive=[ p[0] for p in positive_pattern ]
	
	print('read negative patterns')
	negative=[ p[0] for p in negative_pattern ]

	positive_file_name=osp.join(output_IT,task+'_positive_pattern_tuple_result.txt')
	negative_file_name=osp.join(output_IT,task+'_negative_pattern_tuple_result.txt')
	tool.write_ptn_tpl(retrieve_tuples.keys(), positive, retrieve_tuples, pattern2patternidAll, patternsAll, entities, values, positive_file_name,negative_file_name)
	

	
	patternid2pattern_path= osp.join(feature_path,'patternid2pattern.pickle')
	patternid2pattern = pickle.load(open(patternid2pattern_path,'rb')) 
	pattern2patternid_path=osp.join(feature_path,'pattern2patternid.pickle')
	pattern2patternid = pickle.load(open(pattern2patternid_path,'rb')) # pattern2patternid key format: text\tlocation1\tlocation2
	feature1_path=osp.join(feature_path,'feature1.pickle')
	feature1 = pickle.load(open(feature1_path,'rb'))# feature1: dict. key:patternid, value:pattern embedding
	feature2_path=osp.join(feature_path,'feature2_mean.pickle')
	feature2 = pickle.load(open(feature2_path,'rb'))# feature2: dict. key:patternid, value:attribute embedding


	

	for c in range(iterations):
		print('iterations: '+str(c))
		results1=[]
		results2=[]

		all_X, all_Y, train_pattern_id=tool.getXY(positive,negative,pattern2patternid,feature1,feature2)

		all_X, all_Y=tool.del_bad_neg(all_X, all_Y, theta3)
		n_positive_train=sum(all_Y)
		n_negative_train=len(all_Y)-sum(all_Y)
		print('positive training: ' +str(n_positive_train))
		print('negative training: ' +str(n_negative_train))
		testX, test_pattern_id = tool.getX(train_pattern_id,feature1,feature2)

		all_X = normalize(all_X)#cosine distance
		testX = normalize(testX)#cosine distance

		predict_proba= tool.knn_all(all_X, all_Y, testX, weight='distance',n_neighor=nn)
		positive_patterns=[]
		positive_pattern_probs=[]
		for i in range(len(test_pattern_id)):
			if predict_proba[i][1]>gold_threshold:
				positive_patterns.append(patternid2pattern[test_pattern_id[i]])
				positive_pattern_probs.append(predict_proba[i][1])
		
		new_tpl_file = osp.join(output_Step, task+str(c)+'_tuple_result.txt')

		print('del_bad_pos')
		new_positive_patterns, new_positive_pattern_probs, new_tuples = tool.del_bad_pos(positive_patterns, positive_pattern_probs, pattern2patternidAll, retrieve_tuples, seed_tpl, new_tpl_file, entities,values) #new tuples: id
		new_negative_patterns = []
		print('del_bad_pos done')
		for i in range(len(test_pattern_id)):
			if patternid2pattern[test_pattern_id[i]] in new_positive_patterns:
				results1.append({
				'label': 1,
				'feature': c,
				'pattern': patternid2pattern[test_pattern_id[i]],
				'predict_prob':predict_proba[i][1]})
		
			else:
				new_negative_patterns.append(patternid2pattern[test_pattern_id[i]])
				results2.append({
				'label':0,
				'feature': c,
				'pattern': patternid2pattern[test_pattern_id[i]],
				'predict_prob':predict_proba[i][1]})

		json.dump(results1, open(osp.join(output_Step, task+str(c)+'_positive_pattern'+'.json'), 'w'), indent=1)
		json.dump(results2, open(osp.join(output_Step, task+str(c)+'_negative_pattern'+'.json'), 'w'), indent=1)
		
		#filter tuple
		beta = 5
		m_tau=1

		all_tple=tool.merge_tuples(all_tple+new_tuples)
		E_list, V_list = tool.findGoldTuples(constraint, all_tple,{}, beta,minimal_tau=m_tau)
		
		test_positive_patterns =[]
		for p in new_positive_patterns:
			
			p_id=pattern2patternidAll[p]
			test_positive_patterns.append([p,tool.precisionPatternClose(constraint, E_list, retrieve_tuples[p_id]),patternSummary[p_id][0]])
			if test_positive_patterns[-1][1]>0.5:
				positive.append(p)
		
		test_negative_patterns =[]
		for p in new_negative_patterns:
			p_id=pattern2patternidAll[p]
			test_negative_patterns.append([p,tool.precisionPatternClose(constraint, E_list, retrieve_tuples[p_id]),patternSummary[p_id][0]])
			if test_negative_patterns[-1][1]>0.5:
				positive.append(p)
		
		seed_tpl=tool.writeResult(entities, values, E_list,path=osp.join(output_Test,task+str(c)+'_tuple_result.txt'))#[entityid, valueid, weight]
		positive_file_name=osp.join(output_Test,task+str(c)+'_positive_pattern_tuple_result.txt')
		negative_file_name=osp.join(output_Test,task+str(c)+'_negative_pattern_tuple_result.txt')
		ids=test_pattern_id + train_pattern_id
		tool.write_ptn_tpl(ids, positive, retrieve_tuples, pattern2patternidAll, patternid2pattern, entities, values, positive_file_name,negative_file_name)
