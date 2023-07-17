# Recreates SpliceBERT Results and Runs Small Scale SimCSP

## TODO: 
- implement evaluate.sh
    - implement eval_F1.py
    - implement eval_summarize.py
    - implement eval_plot.py
    - should save a csv to in results with [pretrain:sccs/f1,finetune:sccs/f1,f1]
    - should use spliceator_split to get 5-fold cross-validated results 
- generate baseline for SpliceBERT-human using all metrics 
- start implementation of exp. 2

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
- exp. 1

## NOTES: 
- cite UMAP and leiden algorithm + give more explanation on generation
- hrg uses hg38 and umap plotting uses hg19 but K562 uses hg38 so can only be used w/ msg
    - if pre-training on msg then should use spliceator test set for NMI evaluation
- spliceator num_samples 44,152
- until SpliceBERT-human MLM is implemented correctly, might be learning from hrg during CL 

## IDEAS: 
- experiment with training the classification head with frozen transformer layers first
- use a splice site location tool to generate a better MLM training set
- use a MLM and CL loss function together during pre-train
- explore Sup. CL