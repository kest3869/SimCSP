# Recreates SpliceBERT Results and Runs Small Scale SimCSP

## TODO: 
- create exp. 2 results plots
- update report with exp. 2 results 
- implement remaining metrics 
- evaluate experiment 2 on remaining metrics 
- re-run exp. 1 (structure similar to exp. 2)
- evaluate exp. 2 on all metrics 
- update report with results of exp. 2
- begin work on exp. 3

## EXPERIMENTS: 
- 1: vary learning rate and batch size during pre-training
- 2: vary the amount of pre-training done with a CL loss function 
- 3: chop off k layers and then pre-train a CL function (need to implement)

## METRICS:
- SSP auc and f1 (fine-tune only)
- MLM accuracy (need to implement and pre-train only)
- SIM (need to implement fine-tune)
- NMI (need to implement fine-tune) 

## COMPUTE PRIORITIES: 
- re-run exp. 1 (MEDIUM) 

## NOTES: 
- cite UMAP and leiden algorithm + give more explanation on generation

## IDEAS: 
- explore Sup. CL
- use a MLM and CL loss function together during pre-train