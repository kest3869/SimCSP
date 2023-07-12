# Recreates SpliceBERT Results and Runs Small Scale SimCSP

## TODO: 
- finish pipeline.sh 
    - global 5 fold cross-validation
        - split_spliceator takes 3 args, for_train, fold_num, seed=42
        - it handles splitting and folding the dataset and returns the correct subset 
        - new split should be (70/10/20) (train/val/test)
    - update embed_gen
        - add checkpoint system
    - re-implement scoring evaluation functions 
        - should save a csv to in results with [pretrain:sccs/f1,finetune:sccs/f1,f1]
        - should use spliceator_split to get 5-fold cross-validated results 

## EXPERIMENTS: 
- 1: relationship between f1/sccs in pretrain and fine-tune over time steps=[0,1000, ...]
    - how is pre-training affecting fine-tuning
- 2: relationship between f1/sccs in fine-tune and f1_score over time epochs=[0,1,2, ...]
    - how is fine-tuning affecting model performance 

## METRICS:
- SCCS : spearman correlation w/ cosine similarity 
- NMI : normalized mutual information 

## COMPUTE PRIORITIES: 
- exp. 1
- exp. 2

## NOTES: 
- cite UMAP and leiden algorithm + give more explanation on generation
- hrg uses hg38 and umap plotting uses hg19 but K562 uses hg38 so can only be used w/ msg
- spliceator num_samples 44,152

## IDEAS: 
- explore Sup. CL
- use a MLM and CL loss function together during pre-train
- experiment with varied sequence length during contrastive learning
- experiment with training the classification head with frozen transformer layers 
