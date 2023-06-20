# Adapted from Author Ken Chen 

NN_DICT = {
    'N': 0, 'n': 0,
    'A': 1, 'a': 1,
    'C': 2, 'c': 2,
    'G': 3, 'g': 3,
    'T': 4, 't': 4,
    'U': 4, 'u': 4,
}

def _encode_sequence(seq, nn_dict=NN_DICT):
    ids = []
    seq_len = len(seq)

    for i in range(seq_len):
        ids.append(nn_dict[seq[i]])
    return ids
