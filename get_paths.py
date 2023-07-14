import os

# given an OUT_DIR, return the paths to all models in form: pretrain_paths, finetune_paths
def get_paths(OUT_DIR):
    model_paths = []
    PRETRAIN = os.path.join(OUT_DIR, "pretrained_models/checkpoints/")
    temp_paths = []
    for dirpath, dirnames, _ in os.walk(PRETRAIN):
        for dirname in dirnames:
            folder_path = os.path.join(dirpath, dirname)
            if "Pooling" not in dirname:
                temp_paths.append(folder_path)
    model_paths.append(temp_paths)
    pretrain_paths = model_paths[0]
    pretrain_paths = sorted(pretrain_paths, key=lambda x: int(os.path.basename(x)))
    FINETUNE = os.path.join(OUT_DIR, "finetuned_models/")
    temp_paths = []
    finetune_paths = []
    for dirpath, dirnames, _ in os.walk(FINETUNE):
        for dirname in dirnames:
            folder_path = os.path.join(dirpath, dirname)
            if "epoch" in dirname:
                temp_paths.append(folder_path)
    finetune_paths = sorted(temp_paths)
    return pretrain_paths, finetune_paths
