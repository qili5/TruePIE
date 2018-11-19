import csv
import math
import numpy as np
from io import open
import os.path as osp
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

def readMetaPAD(file_o, file_e_dict='', file_v_dict=''):
	thefile = open(file_o,encoding="utf8")
	#use entity dictionary and value dictionary for entity matching
	if file_e_dict:
		e_line = open(file_e_dict,'r').readlines()
		list_record = [[] for line in e_line]
		e_dict = dict()
		i=0
		for line in e_line:
			list_record[i] = line.rstrip('\n').split('\t')
			for j in range(len(list_record[i])):
				e_dict[list_record[i][j]]=list_record[i][0]        
			i=i+1
	if file_v_dict:
		v_line = open(file_v_dict,encoding="utf8").readlines()
		list_record = [[] for line in v_line]
		v_dict = dict()
		i=0
		for line in v_line:
			list_record[i] = line.rstrip('\n').split('\t')
			for j in range(len(list_record[i])):
				v_dict[list_record[i][j]]=list_record[i][0]        
			i=i+1
#    print(lines[1])

	retrieve_tuples = dict()
	pattern_summary = dict()
	
	patterns,pattern2patternid = [],{} # pattern index
	entities,entity2entityid = [],{} # entity index
	values,value2valueid = [],{} # value index
	
	for line in thefile:
		arr = line.rstrip('\n').split('\t')
		pattern = arr[0]+'\t'+arr[1]+'\t'+arr[2]

		if file_e_dict and arr[3] in e_dict.keys():
			arr[3]=e_dict[arr[3]]
		if file_v_dict and arr[4] in v_dict.keys():
			arr[4]=v_dict[arr[4]]
			
		entity,value,count = arr[3],arr[4],int(arr[5])
		
		if not pattern in pattern2patternid:
			patterns.append(pattern)
			pattern2patternid[pattern] = len(pattern2patternid)
			pattern_summary[pattern]=1
		else:
			pattern_summary[pattern]+=1
		if not entity in entity2entityid:
			entities.append(entity)
			entity2entityid[entity] = len(entity2entityid)
		if not value in value2valueid:
			values.append(value)
			value2valueid[value] = len(value2valueid)
		patternid = pattern2patternid[pattern]
		entityid,valueid = entity2entityid[entity],value2valueid[value]
#        patternid_entityid_valueid_count.append([patternid,entityid,valueid,count])
		
		if patternid not in retrieve_tuples:
			retrieve_tuples[patternid] =[ [entityid,valueid,patternid,count],]
		else:
			retrieve_tuples[patternid].append([entityid,valueid,patternid,count])
		
		
	thefile.close()
	pattern_summary_list=pattern_summary.items()
	pattern_summary_sorted=pattern_summary_list.sort(key=lambda x:x[1],reverse=True)
	return retrieve_tuples, pattern_summary_sorted, patterns,pattern2patternid, entities,entity2entityid, values,value2valueid
	
def find_most_freq(ptn_lst, keyword):
	for ptn in ptn_lst:
		if keyword in ptn[0]:
			return ptn[0]

def constraintTest(tuples,option):
	entity=list(set([l[0] for l in tuples]))
	value=list(set([l[1] for l in tuples]))
	nV=[]
	nE=[]
	for e in entity:
		nV.append(len(set([t[1] for t in tuples if t[0] == e])))
	for v in value:
		nE.append(len(set([t[0] for t in tuples if t[1] == v])))
	if option == '2080':
		i=0
		portion = 0.0
		while portion<0.8:
			i += 1
			unV = len([m for m in nV if m==i])
			portion += unV/len(nV) 

		degree_E = i
	
		i=0
		portion = 0.0
		while portion<0.8:
			i += 1
			unE = len([m for m in nE if m==i])
			portion += unE/len(nE) 

		degree_V = i
	if option == 'median':
		degree_E = math.floor(np.median(nV))
		degree_V = math.floor(np.median(nE))
	if option == 'mode':
		from scipy.stats import mode
		degree_E = math.floor(mode(nV).mode)
		degree_V = math.floor(mode(nE).mode)
	if option == 'mean':
		degree_E = round(np.mean(nV))
		degree_V = round(np.mean(nE))        
	
	i=0
	portion = 0.0
	while portion<0.9:
		i += 1
		unV = nV.count(i)
		portion += unV/len(nV) 


	if i-degree_E>degree_E:
		Ce=str(degree_E) + ' soft'
	else:
		Ce=str(degree_E) + ' hard'
	j=0
	portion = 0.0
	while portion<0.9:
		j += 1
		unE = nE.count(j)
		portion += unE/len(nE) 
	if j-degree_V>degree_V:
		Cv=str(degree_V) + ' soft'
	else:
		Cv=str(degree_V) + ' hard'
	   
	constraint = Ce + '-' + Cv
		
	return constraint

def findGoldTuples(constraint, tuples, rho, beta, minimal_tau):
	bi_graph = []
	edge_weight = []
	
	if rho:
		for e, v, p, count in tuples:
			if [e,v] in bi_graph:
				 edge_weight[bi_graph.index([e,v])] += rho[p][0]*count
			else:
				bi_graph.append([e,v])
				edge_weight.append(rho[p][0]*count)
	else:
		for e, v, count in tuples:
			bi_graph.append([e,v])
			edge_weight.append(count)
			
	Ce = constraint.split('-')[0]
	degree_E = float (Ce.split(' ')[0])
	E_sh = Ce.split(' ')[1]
	Cv = constraint.split('-')[1]
	degree_V = float (Cv.split(' ')[0]) 
	V_sh = Cv.split(' ')[1]
	
	t_count=sorted(zip(edge_weight, bi_graph), reverse=True)
	 
	E_list = dict()
	V_list = dict()


#    t_set = [[l[0], l[1]] for l in tuples]
	
	for w, t in t_count:
#        if t_set.count(t)>100:
		if (w>beta and E_sh=='soft' and V_sh == 'soft'):
			if t[0] in E_list:
				E_list[t[0]][0].append(t[1])
				E_list[t[0]][1].append(w)
			else:
				E_list[t[0]] = [[t[1]], [w]]
			if t[1] in V_list:
				V_list[t[1]][0].append(t[0])
				V_list[t[1]][1].append(w)
			else:
				V_list[t[1]] = [[t[0]], [w]]
		elif w>beta and E_sh=='soft' and V_sh == 'hard':
			if (t[1] in V_list) and (len(V_list[t[1]][0])<degree_V) :
				V_list[t[1]][0].append(t[0])
				V_list[t[1]][1].append(w)
				if t[0] in E_list:
					E_list[t[0]][0].append(t[1])
					E_list[t[0]][1].append(w)
				else:
					E_list[t[0]] = [[t[1]], [w]]
			elif t[1] not in V_list:
				V_list[t[1]] = [[t[0]], [w]]
				if t[0] in E_list:
					E_list[t[0]][0].append(t[1])
					E_list[t[0]][1].append(w)
				else:
					E_list[t[0]] = [[t[1]], [w]]
		elif w>beta and E_sh=='hard' and V_sh == 'soft':
			if (t[0] in E_list) and len(E_list[t[0]][0])<degree_E:
				E_list[t[0]][0].append(t[1])
				E_list[t[0]][1].append(w)
				if t[1] in V_list:
					V_list[t[1]][0].append(t[0])
					V_list[t[1]][1].append(w)
				else:
					V_list[t[1]] = [[t[0]], [w]]
			elif t[0] not in E_list:
				E_list[t[0]] = [[t[1]], [w]]
				if t[1] in V_list:
					V_list[t[1]][0].append(t[0])
					V_list[t[1]][1].append(w)
				else:
					V_list[t[1]] = [[t[0]], [w]]
		elif w>minimal_tau:
			if (t[0] not in E_list) & (t[1] not in V_list):
				E_list[t[0]] = [[t[1]], [w]]
				V_list[t[1]] = [[t[0]], [w]]
			elif t[0] not in E_list:
				if len(V_list[t[1]][0])<degree_V:
					E_list[t[0]] = [[t[1]], [w]]
					V_list[t[1]][0].append(t[0])
					V_list[t[1]][1].append(w)
			elif t[1] not in V_list:
				if len(E_list[t[0]][0])<degree_E:
					V_list[t[1]] = [[t[0]], [w]]
					E_list[t[0]][0].append(t[1])
					E_list[t[0]][1].append(w)
			else:
				if (len(V_list[t[1]][0])<degree_V) & (len(E_list[t[0]][0])<degree_E):
					V_list[t[1]][0].append(t[0])
					V_list[t[1]][1].append(w)
					E_list[t[0]][0].append(t[1])
					E_list[t[0]][1].append(w)
			
	return E_list, V_list
def precisionPattern(constraint, E_list, V_list,tuples_p):
	Ce = constraint.split('-')[0]
	degree_E = float (Ce.split(' ')[0])
#    E_sh = Ce.split(' ')[1]
	Cv = constraint.split('-')[1]
	degree_V = float (Cv.split(' ')[0]) 
#    V_sh = Cv.split(' ')[1]
	fp=0.0
	tp=0.0
	for t in tuples_p:
		if (t[0] in E_list) and (t[1] in E_list[t[0]][0]):        
			tp += t[-1]
		elif ((t[0] in E_list) and (t[1] not in E_list[t[0]][0]) and (len(E_list[t[0]][0]) >= degree_E)) or ((t[1] in V_list) and (t[0] not in V_list[t[1]][0]) and (len(V_list[t[1]][0]) >= degree_V)):  
			fp += t[-1]
		else:
			tp += 0.5*t[-1]
			fp += 0.5*t[-1]
	precision_p=tp/(tp+fp)
	return precision_p

def countTuple(r_tuples):
	tuples_p=set()
	n_tuple =0
	for t in r_tuples:
		tuples_p.add((t[0],t[1],t[2]))
		n_tuple+=t[-1]
	return len(tuples_p),n_tuple
def changeKey_Elist (key_dict, value_dict, old_dict):
	K=list(old_dict.keys())
	for k in K:
		for i in range(len(old_dict[k][0])):
			old_dict[k][0][i]=value_dict[old_dict[k][0][i]]            
		old_dict[key_dict[k]] = old_dict.pop(k)     
	return old_dict

def TruePIE (output_IT, task, retrieve_tuples,patternSummary,gold_patterns, patterns, pattern2patternid, c1, alpha,alpha2, beta, option, constraint='', minimal_tau=2):
	gold_pattern = set()
	for p in gold_patterns:
		if p in pattern2patternid:
			print('Seed Pattern: ' + p)
			gold_pattern.add(pattern2patternid[p])


	precision_pattern=dict()  
	tuples = []
	for p in gold_pattern:
		if patternSummary:
			precision_pattern[p]=[beta/2,patternSummary[p]]
		else:
			u_tuple, n_tuple = countTuple(retrieve_tuples[p])
			
			precision_pattern[p]=[beta/2,u_tuple,n_tuple]
		tuples += retrieve_tuples[p]
		print(p)
		print(precision_pattern[p][1])
	
	if not constraint:    
		constraint = constraintTest(tuples, option)
	E_list, V_list = findGoldTuples(constraint, tuples, precision_pattern,beta,minimal_tau)
	
	all_pattern_set=set(range(len(pattern2patternid)))
	for p in all_pattern_set-gold_pattern:
		if patternSummary:
			precision_pattern[p]=[precisionPattern(constraint, E_list, V_list,retrieve_tuples[p]),patternSummary[p]]
		else:
			u_tuple, n_tuple = countTuple(retrieve_tuples[p])         
			precision_pattern[p]=[precisionPattern(constraint, E_list, V_list,retrieve_tuples[p]),u_tuple,n_tuple]
	
	dist=0
	max_dist=0
	add_pattern = set()
	negative_pattern = set()
	for p in all_pattern_set-gold_pattern:
#    for p in all_pattern_set-gold_pattern:
		dist=max([0, precision_pattern[p][0]-math.pow(c1, precision_pattern[p][1])])
		if dist > max_dist:
			max_dist = dist
		if dist >alpha:            
			add_pattern.add(p)
		neg=precision_pattern[p][0]+math.pow(c1, precision_pattern[p][1]) 
		if neg < alpha2:
			negative_pattern.add(p)
	f3=open(osp.join(output_IT, task+'_step.txt'), 'w')
	i=0
	
	f3.write("*****************Constraint*************************\n")
	f3.write("%s\n" % constraint)
	f3.write("%s\n" % str(i))
	print("constraint: " + constraint)
	f3.write("*****************number of entities*************************\n")
	f3.write("%s\n" % len(E_list))        
	f3.write("*****************number of values*************************\n")
	f3.write("%s\n" % len(V_list)) 
	print("number of entities: " + str(len(E_list)))
	print("number of values: " + str(len(V_list)))
	
	while max_dist>alpha:

		i+=1
		
		f3.write("*****************number of added patterns*************************\n")
		f3.write("%s\n" % len(add_pattern))

		f3.write("*****************added patterns*************************\n")
		for p in add_pattern:
			f3.write("%s\n" % patterns[p])                
		
		
		gold_pattern.update(add_pattern)
		print("number of gold patterns: "+ str(len(gold_pattern)))
	   
		f3.write("*****************number of gold patterns*************************\n")
		f3.write("%s\n" % len(gold_pattern))
		
		
		f3.write("*****************number of bad patterns*************************\n")
		f3.write("%s\n" % len(negative_pattern))
		f3.write("*****************bad patterns*************************\n")
		for p in negative_pattern:
			f3.write("%s\n" % patterns[p])
			

#        for p in add_pattern:
#            add_tuples = retrieve_tuples[p]
#            tuples+=add_tuples
		tuples=[]
		for p in gold_pattern:
			add_tuples = retrieve_tuples[p]
			tuples+=add_tuples
		if not constraint: 
			constraint = constraintTest(tuples, option)
		E_list, V_list = findGoldTuples(constraint, tuples, precision_pattern,beta,minimal_tau)
		max_dist=0
		add_pattern = set()
		negative_pattern=set()
#        for p in all_pattern_set-gold_pattern:        
		for p in all_pattern_set-gold_pattern:
			precision_pattern[p][0]=precisionPattern(constraint, E_list, V_list,retrieve_tuples[p])
			dist=max([0, precision_pattern[p][0]-math.pow(c1, precision_pattern[p][1])])
			if dist > max_dist:
				max_dist = dist
			if dist >alpha:
				add_pattern.add(p)
			neg=precision_pattern[p][0]+math.pow(c1, precision_pattern[p][1]) 
			if neg < alpha2:
				negative_pattern.add(p)
		
		f3.write("*****************Constraint*************************\n")
		f3.write("%s\n" % constraint)
		f3.write("%s\n" % str(i))
		print("current iteration" + str(i))
		print("constraint: " + constraint)
		f3.write("*****************number of entities*************************\n")
		f3.write("%s\n" % len(E_list))        
		f3.write("*****************number of values*************************\n")
		f3.write("%s\n" % len(V_list)) 
		print("number of entities: " + str(len(E_list)))
		print("number of values: " + str(len(V_list)))
	f3.close()
	all_pattern_precision = []
	positive_pattern=[]

	negative_pattern=[]

	for p in all_pattern_set:
		if patterns[p] in gold_patterns:
			precision_pattern[p][0]=1
			dist = 1
		else:
			precision_pattern[p][0]=precisionPattern(constraint, E_list, V_list,retrieve_tuples[p])
			dist=max([0, precision_pattern[p][0]-math.pow(c1, precision_pattern[p][1])])
		if dist >alpha:
			positive_pattern.append([patterns[p],precision_pattern[p][0],precision_pattern[p][1]])
		neg=precision_pattern[p][0]+math.pow(c1, precision_pattern[p][1]) 
		if neg <alpha2:
			negative_pattern.append([patterns[p],precision_pattern[p][0],precision_pattern[p][1]])
			
		all_pattern_precision.append([patterns[p],precision_pattern[p][0],precision_pattern[p][1]])
	
	
	print("number of gold patterns: "+ str(len(positive_pattern)))
	positive_pattern.sort(key=lambda x:x[1],reverse=True)   
	negative_pattern.sort(key=lambda x:x[1])
	all_pattern_precision.sort(key=lambda x:x[1],reverse=True)  
	
	with open(osp.join(output_IT, task+'_positive_pattern_result.txt'), 'w',newline='') as f:
		w = csv.writer(f, dialect='excel-tab')
		w.writerows(positive_pattern)        
	with open(osp.join(output_IT, task+'_negative_pattern_result.txt'), 'w',newline='') as f:
		w = csv.writer(f, dialect='excel-tab')
		w.writerows(negative_pattern)    
	with open(osp.join(output_IT, task+'_all_pattern_result.txt'), 'w',newline='') as f:
		w = csv.writer(f, dialect='excel-tab')
		w.writerows(all_pattern_precision)
	
	return E_list, positive_pattern, negative_pattern, constraint

def writeResult(entities, values, E_list,path):
	tuple_result1=[]
	for e in E_list:
		for i in range(len(E_list[e][0])):
			tuple_result1.append([e, E_list[e][0][i], E_list[e][1][i]])
	E_list_copy=dict(E_list)
	E_list_new = changeKey_Elist (entities, values, E_list_copy)
	tuple_result=[]
	for e in E_list_new:
		for i in range(len(E_list_new[e][0])):
			tuple_result.append([e, E_list_new[e][0][i], E_list_new[e][1][i]])
	
	with open(path, 'w',newline='') as f:
		w = csv.writer(f, dialect='excel-tab')
		w.writerows(tuple_result)
	return tuple_result1
	
def positiveP2T(patterns, ptn2ptnid, ptn2tpl, min_p=0.2):
	tuples=[]
	for p in patterns:
		if p[1]<min_p:
			continue
		tpl_list = ptn2tpl[ptn2ptnid[p[0]]]
		for t in tpl_list:
			tuples.append([t[0],t[1],t[3]])
	return tuples
	
def write_ptn_tpl(id_list,positives, ptn2tpl, ptn2ptnid_2,ptnid2ptn, etyid2ety, valid2val, p_file_name, n_file_name):
	output_rows_p = []
	output_rows_n = []
	for id in id_list:
		if ptnid2ptn[id] in positives:
			p=ptnid2ptn[id]
			output_rows_p.append([p])
			tpl_list = ptn2tpl[ptn2ptnid_2[p]]
			tpl_list.sort(key=lambda tpl: tpl[2], reverse=True)
			for tpl in tpl_list:
				output_rows_p.append([''] + [etyid2ety[tpl[0]], valid2val[tpl[1]], tpl[3]])
		else:
			p=ptnid2ptn[id]
			output_rows_n.append([p])
			tpl_list = ptn2tpl[ptn2ptnid_2[p]]
			tpl_list.sort(key=lambda tpl: tpl[2], reverse=True)
			for tpl in tpl_list:
				output_rows_n.append([''] + [etyid2ety[tpl[0]], valid2val[tpl[1]], tpl[3]])
	with open(p_file_name, newline='', mode='w') as csv_file:
		csv_writer = csv.writer(csv_file, dialect='excel-tab')
		csv_writer.writerows(output_rows_p)
	with open(n_file_name, newline='', mode='w') as csv_file:
		csv_writer = csv.writer(csv_file, dialect='excel-tab')
		csv_writer.writerows(output_rows_n)

def pattern2features(f1, f2={}, pattern=[], pattern2patternid=[]):
	X=[]
	pattern_id=[]
	if pattern:
		for p in pattern:
			if p in pattern2patternid:
				patternID = pattern2patternid[p]
				pattern_id.append(patternID)
			else:
				continue
			if f2:
				X.append(f1[patternID]+f2[patternID])
			else:
				X.append(f1[patternID])
		X=np.array(X)
	else:
		for p in f1:
			pattern_id.append(p)
			if f2:
				X.append(f1[p]+f2[p])
			else:
				X.append(f1[p])
	return X, pattern_id

def getXY(positive,negative,pattern2patternid,feature1,feature2={}):
	print('get positive training samples')
	positive_X,positive_pattern_id = pattern2features(feature1,feature2,positive,pattern2patternid)
	# t_p_X = positive_X[1001:]
	# t_p_Y = np.array([1] * len(t_p_X))
	#positive_X = positive_X[:1001]
	positive_Y = np.array([1] * len(positive_X))
	print ('number of positive samples:' + str(len(positive_X)))
	
	negative_X, negative_pattern_id = pattern2features(feature1,feature2,negative,pattern2patternid)
	# t_n_X = negative_X[1001:]
	# t_n_Y = np.array([0] * len(t_n_X))
	#negative_X = negative_X[:1001]
	negative_Y = np.array([0] * len(negative_X))
	print('number of negative samples:' + str(len(negative_X)))
	
	all_X=np.concatenate([positive_X,negative_X],axis=0)
	all_Y=np.concatenate([positive_Y,negative_Y],axis=0)
	pattern_id=np.concatenate([positive_pattern_id,negative_pattern_id],axis=0).tolist()
	# test_X=np.concatenate([t_p_X,t_n_X],axis=0)
	# test_Y=np.concatenate([t_p_Y,t_n_Y],axis=0)
	print('size of data: ' + str(len(all_X)))
	return all_X, all_Y, pattern_id

def del_bad_neg(embeddings, labels, threshold):
	"""
	Delete a negative pattern if its embedding is similar enough to that of any of the positive patterns
	:param embeddings: 2-d numpy array
	:param labels: 2-d numpy array
	:param pattern_ids: list of ints
	:param threshold: int
	:return: 2-d numpy array, 2-d numpy array, list of ints
	"""
	labels2 = [True if a == 1 else False for a in labels]
	pos = embeddings[labels2]
	retain_lst = []
	for i, emb in enumerate(embeddings):
		if not labels2[i]:
			kernel_matrix = cosine_similarity(np.array([emb]), pos)
			if np.amax(kernel_matrix[0]) > threshold:
				retain_lst.append(False)
				continue
		retain_lst.append(True)
	return embeddings[np.array(retain_lst)], labels[np.array(retain_lst)]

def getX(train_id,feature1,feature2={}):
	test_X_,pattern_id_ = pattern2features(f1=feature1,f2=feature2)
	test_X,pattern_id=[],[]
	if train_id:
		#remove training data
		for i in range(len(pattern_id_)):
			if pattern_id_[i] not in train_id:
				test_X.append(test_X_[i])
				pattern_id.append(pattern_id_[i])
	print('size of unlabeled data: ' + str(len(test_X)))
	return np.array(test_X), pattern_id

def knn_all(all_X, all_Y, testX, n_neighor=5, weight='uniform'):
	print('run knn')
	neigh = KNeighborsClassifier(n_neighbors=n_neighor, n_jobs=40, weights=weight)
	neigh.fit(all_X, all_Y)
	y_predict = neigh.predict_proba(testX)
	return y_predict
	
def del_bad_pos(ptn_list, prob_list, ptn2ptnid, ptn_dict, seed_tpls, new_tpls_file, etyid2ety, valid2val):
	"""
	del_bad_pos(positive_patterns, positive_pattern_probs, pattern2patternidAll, retrieve_tuples, seed_tpl, new_tpl_file, entities,values)
	Delete a positive pattern after classification if the none of the entities/values it extracts is contained in tuples
	extracted by the seeds
	:param ptn_list: list of strings
	:param prob_list: list of floats
	:param ptn_dict: dict whose key is id and value is list of tuples
	:param seed_tpls: list of lists
	:param new_tpls_file: string
	:param etyid2ety: list
	:param valid2val: list
	:return:
	"""

	seed_tpls_pairs = [[t[0],t[1]] for t in seed_tpls ]
	
	retain_lst = []
	new_tuples = []
	for i, ptn in enumerate(ptn_list):
		found = False
		# 2-d numpy array
		tpl_list = ptn_dict[ptn2ptnid[ptn]]
		for tpl in tpl_list:
			#if tpl[0] in seed_tpls[0] or tpl[1] in seed_tpls[1]:
			if [tpl[0],tpl[1]] in seed_tpls_pairs:
				retain_lst.append(True)
				found = True
				break
		if found:
			for tpl in tpl_list:
				tpl[2] = prob_list[i]
			new_tuples.extend(tpl_list)
		else:
			retain_lst.append(False)

	new_tuples_restored1 = merge_tuples(new_tuples)

	new_tuples_restored = restore_str(new_tuples_restored1, etyid2ety, valid2val)

	with open(new_tpls_file, 'w') as csv_file:
		csv_writer = csv.writer(csv_file, delimiter='\t', quoting=csv.QUOTE_NONE,quotechar='',)
		csv_writer.writerows(new_tuples_restored)
	

	return np.array(ptn_list)[np.array(retain_lst)].tolist(), np.array(prob_list)[np.array(retain_lst)].tolist(),new_tuples_restored1

def restore_str(tpl_list, etyid2ety, valid2val):
	tpl_list2 = []
	for tpl in tpl_list:
		tpl_list2.append([etyid2ety[tpl[0]], valid2val[tpl[1]], tpl[2]])
	return tpl_list2



def merge_tuples(tpl_list):
	tpl_list.sort(key=lambda x: x[1])
	tpl_list.sort(key=lambda x: x[0])
	merged_tpl_list = []
	for i, tpl in enumerate(tpl_list):
		if i > 0 and tpl[0] == tpl_list[i - 1][0] and tpl[1] == tpl_list[i - 1][1]:
			merged_tpl_list[-1][2] += tpl[2]
		else:
			merged_tpl_list.append([tpl[0], tpl[1],tpl[2]])
	return merged_tpl_list

def precisionPatternClose(constraint, E_list, tuples_p):
	Ce = constraint.split('-')[0]
	degree_E = float (Ce.split(' ')[0])
#    E_sh = Ce.split(' ')[1]
	Cv = constraint.split('-')[1]
	degree_V = float (Cv.split(' ')[0]) 
#    V_sh = Cv.split(' ')[1]
	fp=0.0
	tp=0.0
	for t in tuples_p:
		if (t[0] in E_list) and (t[1] in E_list[t[0]][0]):        
			#tp += t[-1]
			tp += 1
		else:
			#fp += t[-1]
			fp += 1
	precision_p=tp/(tp+fp)
	return precision_p
	
