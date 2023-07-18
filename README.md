# Recreates SpliceBERT Results and Runs Small Scale SimCSP

## TODO: 
- implement evaluate.sh
    - implement eval_plot.py : should save a csv to in results with [pretrain:sccs/f1,finetune:sccs/f1,f1]
- update train.py pretrain.py, and embed_gen.py to have + '/finetuned_models/' where used instead of += at beginnning of file for consistency and predictability

## EXPERIMENTS: 
- 1: relationship between f1/sccs in pretrain and fine-tune and its affects on F1 score 
    - uses the last layer's [CLS] token 
    - uses SpliceBERT-human, evaluated using f1/sccs on spliceator and NMI on hg19
    - could be improved by using SpliceBERT and NMI on spliceator 
- 2: experiment with chopping off 2 layers and doing CL + fine-tune on that

## METRICS:
- SCCS : spearman correlation w/ cosine similarity 
- NMI : normalized mutual information 

## COMPUTE PRIORITIES:
Exp1:
- compute models for BEST_SCCS
- compute models for BASELINE CL steps
- compute results for BASELINE, BEST_NMI, and BEST_SCCS
Exp2: 
- compute models with 5000k steps w/ and w/o (-1/-2) hidden layers
and exp.2
- compute models with 0 CL steps (-1/-2) hidden layers 
- compute best models for NMI exp. 2

## NOTES: 
- cite UMAP and leiden algorithm + give more explanation on generation
- hrg uses hg38 and umap plotting uses hg19 but K562 uses hg38 so can only be used w/ msg
    - if pre-training on msg then should use spliceator test set for NMI evaluation
- spliceator num_samples 44,152
- until SpliceBERT-human MLM is implemented correctly, might be learning from hrg during CL 

## KNOWN IMPROVEMENTS: 
- bigger model
- more diverse training data
- use middle transformer layers 
- use mean pooling accross some linear combination of hidden layers 

## IDEAS: 
- experiment with training the classification head with frozen transformer layers first
- use a splice site location tool to generate a better MLM training set
- use a MLM and CL loss function together during pre-train
- explore Sup. CL
