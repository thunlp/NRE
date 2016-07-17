Neural relation extraction aims to extract relations from plain text with neural models, which has been the state-of-the-art methods for relation extraction. In this project, we provide our implementations of CNN [Zeng et al., 2014] and PCNN [Zeng et al.,2015] and their extended version with sentence-level attention scheme [Lin et al., 2016] .

Evaluation Results
==========
 ![image](https://github.com/mrlyk423/figure/raw/master/tot.pdf)
 


DATA
==========

We provide NYT10  dataset we used for the task relation extraction in data/ directory. We preprocess the original data to make it satisfy the input format of our codes. The original data of NYT10 can be downloaded from:

Relation Extraction:  NYT10 is originally released by the paper "Sebastian Riedel, Limin Yao, and Andrew McCallum. Modeling relations and their mentions without labeled text." [[Download]]( http://iesl.cs.umass.edu/riedel/ecml/)

Pre-Trained Word Vectors are learned from New York Times Annotated Corpus (LDC Data LDC2008T19), which should be obtained from LDC (https://catalog.ldc.upenn.edu/LDC2008T19).

To run our code, the dataset should be put in the folder data/ using the following format, containing six files

+ train.txt: training file, format (fb_mid_e1, fb_mid_e2, e1_name, e2_name, relation, sentence).

+ test.txt: test file, same format as train.txt.

+ entity2id.txt: all entities and corresponding ids, one per line.

+ relation2id.txt: all relations and corresponding ids, one per line.

+ vec.bin: the pre-train word embedding file

CODE
==========

The source codes of various methods are put in the folders CNN+MAX/, CNN+ATT/, PCNN+MAX/, PCNN+ATT/.

COMPILE 
==========

Just type "make" in the corresponding folders.

TRAIN
==========

For training, you need to type the following command in each model folder:

./train

The training model file will be saved in folder out/ .

TEST
==========

For testing, you need to type the following command in each model folder:

./test

The testing result which reports the precision/recall curve  will be shown in pr.txt.

CITE
==========

If you use the code, please cite the following paper:

[Lin et al., 2016] Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. Neural Relation Extraction with Selective Attention over Instances. ACL2016 .[[pdf]](http://thunlp.org/~lyk/publications/acl2016_nre.pdf)

Reference
==========
[Zeng et al., 2014] Daojian Zeng, Kang Liu, Siwei Lai, Guangyou Zhou, and Jun Zhao. Relation classification via convolutional deep neural network. In Proceedings of COLING.
[Zeng et al.,2015] Daojian Zeng,Kang Liu,Yubo Chen,and Jun Zhao. Distant supervision for relation extraction via piecewise convolutional neural networks. In Proceedings of EMNLP.
