
# Simple Contrastive Splice Site Prediction

## Reader's Note
- Project write up can be found in SimCSP.pdf
- Results and visualizations can be found under Experimental_Results/Exp2

## EXPERIMENTS: 
- 1: relationship between f1/sccs in pretrain and fine-tune and its affects on F1 score 
    - uses the last layer's [CLS] token 
    - uses SpliceBERT-human, evaluated using f1/sccs on spliceator and NMI on hg19
    - could be improved by using SpliceBERT and NMI on spliceator 
- 2 - hyperparameter search over pre-train w/ SCCS validation
- 3: experiment with chopping off 1 layers transformer and doing CL + fine-tune on that
- 4: experiment with chopping off 2 transformer layers and doing CL + fine-tune on that
- 5: freeze the transformer layers and only fine-tune the classifier 
    - this will tell you if the internal representation is being improved 

## METRICS:
- SCCS : spearman correlation w/ cosine similarity 
- NMI : normalized mutual information 
- F1 : linear combination of precision and recall

## EXPERIMENTAL PIPELINE:
- pretrain.sh
- finetune.sh

