# TruePIE
TruePIE: Discovering Reliable Patterns in Pattern-Based Information Extraction

Input

1. pattern-extraction. Pattern generation can be found at: https://github.com/mjiang89/MetaPAD
2. word embedding: word2vec

Input format

1. /input/pattern.txt

PATTERN\tTOKEN_INDEX_OF_ENTITY\tTOKEN_INDEX_OF_VALUE\tENTITY\tVALUE\tCOUNT

Example: $LOCATION leader $PERSON	0	2	united_states	trump	1001

2. /input/word_embedding.pickle or /input/word_embedding.txt

word_embedding.pickle: dictionary, where key is word, value is the vector

word_embedding.txt: the txt output of word2vec tool

Output

  The intermediate results are also provided. The final results can be found in /output/result/

Model Parameters

  Can be changed in run_for_task.py 

Package Requirement:
numpy, json, csv, \_pickle, sklearn

Citation

@inproceedings{li2018truepie,
  title={Truepie: Discovering reliable patterns in pattern-based information extraction},
  author={Li, Qi and Jiang, Meng and Zhang, Xikun and Qu, Meng and Hanratty, Timothy P and Gao, Jing and Han, Jiawei},
  booktitle={Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1675--1684},
  year={2018},
  organization={ACM}
}
