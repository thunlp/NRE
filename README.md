DATA
==========

I provide NYT10  datasets we used for the task relation extraction  with the input format of my code in data/ directory.

The original data use in the experiment can download in:

Relation Extraction:  NYT10 [Riedel et al., 2010] are published bythe paper "Sebastian Riedel, Limin Yao, and Andrew McCallum. Modeling relations and their mentions without labeled text." [[Download]]( http://iesl.cs.umass.edu/riedel/ecml/)

Word vectors pre-training: New York Times Annotated Corpus (LDC Data LDC2008T19). If you want the data, you should buy from LDC (https://catalog.ldc.upenn.edu/LDC2008T19) first.

Datasets are needed in the folder data/ in the following format

Dataset contains six files:



+ train.txt: training file, format (fb_mid_e1, fb_mid_e2, e1_name, e2_name, relation, sentence).

+ test.txt: test file, same format as train.txt.

+ entity2id.txt: all entities and corresponding ids, one per line.

+ relation2id.txt: all relations and corresponding ids, one per line.

+ vec.bin: the pre-train word embedding file




CODE
==========

In the folder CNN+MAX/, CNN+ATT/, PCNN+MAX/, PCNN+ATT/:



COMPILE 
==========

Just type "make" in the model folder



TRAIN
==========

For training, You need follow the step in each model folder:


./train

The training model file will be saved in folder out/


TEST
==========

For testing, You need follow the step in each model folder:

./test

It will evaluate on pr.txt and report the precision/recall curve.




CITE
==========

If you use the code, you should cite the following paper:

Lin, Yankai, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Neural Relation Extraction with Selective Attention over Instances. ACL2016 .[[pdf]](http://thunlp.org/~ssq/publications/acl2016_RE.pdf)