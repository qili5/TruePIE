#!/bin/bash

#step0 word2vec
#step0 metaPAD
#step1 pattern embedding
#step2 self-training
#step2.1 truth discovery
#step2.2 train and predict
#step2.3 new positive patterns
#step2.4 go to step2.2

#step1 get pattern embedding
METAPAD='/input/pattern.txt'
EMBEDDING='/input/word_embedding.pickle' # or '/input/word_embedding.txt'
OUTPUT_PE='/input/feature/'
echo ===get pattern embedding===
if [ -d ${OUTPUT_PE} ];
then 
	echo ===pattern embedding done===
else
	mkdir ${OUTPUT_PE}
	echo ===running===
	python gen_feature_vec.py ${METAPAD} ${EMBEDDING} ${OUTPUT_PE} 
	echo ===pattern embedding done===
fi

#step2 self-training
OUTPUT_IT='/output/initial_training'
OUTPUT_STEP='/output/intermediate'
OUTPUT_TEST='/output/result'

if [ -d ${OUTPUT_IT} ];
then 
	echo ===output directories exists===
else
	echo ===create output directories===
	mkdir ${OUTPUT_IT}
fi

if [ -d ${OUTPUT_STEP} ];
then 
	echo ===output directories exists===
else
	echo ===create output directories===
	mkdir ${OUTPUT_STEP}
fi

if [ -d ${OUTPUT_TEST} ];
then 
	echo ===output directories exists===
else
	echo ===create output directories===
	mkdir ${OUTPUT_TEST}
fi

python run_for_task.py ${METAPAD} ${OUTPUT_PE} ${OUTPUT_IT} ${OUTPUT_STEP} ${OUTPUT_TEST} 