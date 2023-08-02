# libraries
import argparse
import logging
import os
import csv
import numpy as np
import sys 
import datetime
import torch
import torch.nn
from torch.utils.data import Subset 
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# files
import split_spliceator
from get_paths import get_paths
import load

# return a list of SCCS scores and associated file paths 
def get_SCCS_scores(model_paths, ds_eval, fold_num):
    '''
    Inputs: 
    - models : a list of filepaths to trained models 
    - ds_eval : a setence-transformers style evaluation datase for semantic similarity 
    Outputs:
    - sccs_model_paths : the paths to the models that were scored 
    - sccs_scores : a list of scores received by the associated models 
    '''
    
    # initalize lists to hold our data 
    sccs_scores = []
    sccs_model_paths = []

    # get sccs scores for all models in model paths 
    for model_path in model_paths:
        if ("fold" + str(fold_num)) in model_path:
            # initialize the model to be evaluated
            word_embedding_model = models.Transformer(model_path, max_seq_length=400)
            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            # create our evaluator (only using the first fold for sccs)
            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(ds_eval, 
                                                                        batch_size=8, 
                                                                        name='spliceator_test_split',
                                                                        show_progress_bar=True)
            # pass the model through the evaluator 
            sccs = evaluator(model)
            # save the evaluation information
            sccs_scores.append(sccs)
            sccs_model_paths.append(model_path)

    return sccs_model_paths, sccs_scores

if __name__ == "__main__":
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Get_SCCS_scores')
    parser.add_argument('-p', '--out_dir', type=str, help='The model save path')
    parser.add_argument('--for_pretrain', action='store_true', help='Use this flag for pretraining.')
    parser.add_argument('--split_dir', type=str, help='The path to the splits')
    # Parse the command line arguments
    args = parser.parse_args()
    # Retrieve the values of the command line arguments
    OUT_DIR = args.out_dir
    SPLIT_DIR = args.split_dir
    for_pretrain = args.for_pretrain

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(OUT_DIR + '/results/' + 'eval_SCCS.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # skip if already completed 
    if os.path.exists(OUT_DIR + '/results/' + '/finished_SCCS.pt'):
        logger.info("Found finished, skipping SCCS.")
        print("Found finished, skipping eval_SCCS.py!")
        sys.exit()


    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('/storage/store/kevin/data/tokenizer_setup')
    # Positive and Negative paths
    positive_dir = '/storage/store/kevin/data/spliceator/Training_data/Positive/GS'
    negative_dir = '/storage/store/kevin/data/spliceator/Training_data/Negative/GS/GS_1'
    # List all files in the directory
    positive_files = [os.path.join(positive_dir, file) for file in os.listdir(positive_dir)]
    negative_files = [os.path.join(negative_dir, file) for file in os.listdir(negative_dir)]
    # Load dataset using class from load.py file
    ds = load.SpliceatorDataset(
        positive=positive_files,
        negative=negative_files,
        tokenizer=tokenizer,
        max_len=400
    )

    # get num folds 
    num_folds = np.shape(torch.load(SPLIT_DIR + 'validation_split.pt'))[0]

    if for_pretrain:
        # load the split the fine-tuned model used 
        split = torch.load(SPLIT_DIR + '/train_split.pt')
        # get paths to models 
        model_paths, _ = get_paths(OUT_DIR)
    else:
        # load the split the fine-tuned model used 
        split = torch.load(SPLIT_DIR + '/test_split.pt')
        # get paths to models 
        _, model_paths = get_paths(OUT_DIR)

    # calculates the paths and scores 
    paths, scores = [], []
    for i in range(num_folds):
        # call split_spliceator.pre_val_data to build semantic similarity dataset from subset
        ds_prepped = split_spliceator.prep_val_data(Subset(ds, split[i]), tokenizer)
        # get scores 
        paths_temp, scores_temp = get_SCCS_scores(model_paths, ds_prepped, i)
        paths.append(paths_temp)
        scores.append(scores_temp)  # Append scores_temp as well

    # Combine paths and scores for each fold using zip and transpose
    fold_results = list(zip(paths, scores))

    # Save CSV of pretrained scores for each fold
    with open(OUT_DIR + '/results/' + 'SCCS_results.csv', "w", newline="") as file:
        writer = csv.writer(file) 
        writer.writerow(['Path', 'Score'])  # Write header
        for path, score in fold_results:
            for p, s in zip(path, score):
                writer.writerow([p, s])

    # mark evaluation as finished
    torch.save(str(for_pretrain), OUT_DIR + '/results/' + '/finished_SCCS.pt')
