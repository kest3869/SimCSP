
Recreates SpliceBERT Results and Runs Small Scale SimCSP

TODO

load_pretrain.py : uses only validation set of human reference genome
load_spliceator_CL.py : NOT IMPLEMENTED : will convert spliceator dataset to suitable format for CL
pretrain_CL.py : does not use [CLS] token for classification
train.py : does not support CL OR multi-gpu training
test.py : does not support 10 fold cross validation from SpliceBERT
pretrain_MLM.py : NOT IMPLEMENTED : will train MLM for same number of epochs as CL for comparison
build github repo
upload simcsp docker image
document dataset downloading and directory organization scheme