import torch
import numpy as np

BLOSUM50_MATRIX = np.array([
    [5, -2, -1, -2, -1, -3, 0, -2, -1, -2, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
    [-2, 7, -1, -2, -4, 1, -1, -3, 0, -4, -3, 3, -2, -2, -3, -1, -1, -3, -1, -3],
    [-1, -1, 7, 2, -2, 0, 0, 0, 1, -3, -4, 0, -2, -4, -2, 1, 0, -4, -2, -3],
    [-2, -2, 2, 8, -4, 0, 2, -1, -1, -4, -4, -1, -4, -5, -1, 0, -1, -5, -3, -4],
    [-1, -4, -2, -4, 11, -3, -4, -3, -4, -5, -5, -3, -5, -4, -3, -1, -1, -4, -4, -1],
    [-3, 1, 0, 0, -3, 7, -1, -2, 1, -2, -1, 2, 0, -3, -2, 0, -1, -1, -1, -2],
    [0, -1, 0, 2, -4, -1, 7, -2, -1, -4, -3, 0, -2, -4, -1, 0, -1, -4, -3, -3],
    [-2, -3, 0, -1, -3, -2, -2, 5, -2, -3, -3, -2, -3, -3, -2, 0, -2, -3, -3, -4],
    [-1, 0, 1, -1, -4, 1, -1, -2, 5, -3, -2, 1, 0, -3, -1, 0, -1, -3, -2, -3],
    [-2, -4, -3, -4, -5, -2, -4, -3, -3, 5, 2, -3, 2, 0, -3, -3, -2, -2, 1, 4],
    [-1, -3, -4, -4, -5, -1, -3, -3, -2, 2, 5, -2, 3, 1, -4, -3, -2, -2, 0, 1],
    [-1, 3, 0, -1, -3, 2, 0, -2, 1, -3, -2, 6, -2, -4, -1, 0, -1, -3, -2, -3],
    [-1, -2, -2, -4, -5, 0, -2, -3, 0, 2, 3, -2, 7, 0, -4, -2, -1, -1, 1, 2],
    [-2, -2, -4, -5, -4, -3, -4, -3, -3, 0, 1, -4, 0, 8, -3, -2, -2, 4, 4, 0],
    [-1, -3, -2, -1, -3, -2, -1, -2, -1, -3, -4, -1, -4, -3, 9, -1, -1, -3, -3, -3],
    [1, -1, 1, 0, -1, 0, 0, 0, 0, -3, -3, 0, -2, -2, -1, 5, 2, -4, -2, -1],
    [0, -1, 0, -1, -1, -1, -1, -2, -1, -2, -2, -1, -1, -2, -1, 2, 5, -3, -2, 0],
    [-3, -3, -4, -5, -4, -1, -4, -3, -3, -2, -2, -3, -1, 4, -3, -4, -3, 15, 2, -1],
    [-2, -1, -2, -3, -4, -1, -3, -3, -2, 1, 0, -2, 1, 4, -3, -2, -2, 2, 8, 0],
    [0, -3, -3, -4, -1, -2, -3, -4, -3, 4, 1, -3, 2, 0, -3, -1, 0, -1, 0, 5]
])

def one_hot_encoding(seq):
    # List of 20 standard amino acids.
    amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                   'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    # Mapping from amino acids to indices
    amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    # Length of the protein sequence
    seq_len = len(seq)
    # Initialize an empty one-hot encoded matrix
    encoded_seq = np.zeros((seq_len, 20))
    # Encode the protein sequence
    for i, aa in enumerate(seq):
        if aa in amino_acids:
            encoded_seq[i, amino_acid_to_index[aa]] = 1

    return encoded_seq

def bolossum_encoding(seq, matrix):
    # List of 20 standard amino acids.
    amino_acids = "ARNDCQEGHILKMFPSTWYV"
    # Mapping from amino acids to indices
    amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    # Length of the protein sequence
    seq_len = len(seq)
    # Initialize an empty one-hot encoded matrix
    encoded_seq = np.full((seq_len, 20), -1)
    # Encode the protein sequence
    for i, aa in enumerate(seq):
        if aa in amino_acids:
            encoded_seq[i] = matrix[amino_acid_to_index[aa]]

    return encoded_seq

def atchley_factor_encoding(seq):
    atchley_factors = {
        'A': [-0.591, -1.302, -0.733, 1.570, -0.146],
        'C': [-1.343, 0.465, -0.862, -1.020, -0.255],
        'D': [1.050, 0.302, -3.656, -0.259, -3.242],
        'E': [1.357, -1.453, 1.477, 0.113, -0.837],
        'F': [-1.006, -0.590, 1.891, -0.397, 0.412],
        'G': [-0.384, 1.652, 1.330, 1.045, 2.064],
        'H': [0.336, -0.417, -1.673, -1.474, -0.078],
        'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
        'K': [1.831, -0.561, 0.533, -0.277, 1.648],
        'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
        'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
        'N': [0.945, 0.828, 1.299, -0.169, 0.933],
        'P': [0.189, 2.081, -1.628, 0.421, -1.392],
        'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
        'R': [1.538, -0.055, 1.502, 0.440, 2.897],
        'S': [-0.228, 1.399, -4.760, 0.670, -2.647],
        'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
        'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
        'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
        'Y': [0.260, 0.830, 3.097, -0.838, 1.512]
    }

    # List of amino acids
    amino_acids = list(atchley_factors.keys())
    # Mapping from amino acids to indices
    amino_acid_to_index = {aa: i for i, aa in enumerate(amino_acids)}
    # Length of the protein sequence
    seq_len = len(seq)
    # Number of Atchley factors
    num_factors = len(atchley_factors[amino_acids[0]])
    # Initialize an empty matrix to store the encoded sequence
    encoded_seq = np.zeros((seq_len, num_factors))
    # Encode the protein sequence
    for i, aa in enumerate(seq):
        if aa in amino_acids:
            encoded_seq[i] = atchley_factors[aa]

    return encoded_seq

def pad_protein_sequence(sequence, max_seq_len, pad_token='X'):
    original_length = len(sequence)
    
    padded_sequence = sequence.ljust(max_seq_len, pad_token)
    
    padded_sequence = padded_sequence[:max_seq_len]
    
    mask = [1] * original_length + [0] * (max_seq_len - original_length)
    mask = mask[:max_seq_len]

    return padded_sequence, mask

def mhc_encoding(mhc_seq, max_seq_len):
    # max_seq_len=366 for MHC(HLA) class I
    # max_seq_len=34 for pseudo seq
    seq, mask = pad_protein_sequence(mhc_seq, max_seq_len)

    one_hot_feat = one_hot_encoding(seq)
    blosum50_feat = bolossum_encoding(seq, BLOSUM50_MATRIX)
    atchley_factor_feat = atchley_factor_encoding(seq)

    concat_feat = np.concatenate(
        (one_hot_feat, blosum50_feat, atchley_factor_feat), axis=1)
    return concat_feat, mask

def epitope_encoding(epitope_seq, max_seq_len):
    # max_seq_len=15 for epitope
    seq, mask = pad_protein_sequence(epitope_seq, max_seq_len)

    one_hot_feat = one_hot_encoding(seq)
    return one_hot_feat, mask

def tcra_fv_encoding(tcra_fv, max_seq_len):
    # max_seq_len = ? for tcra_fv
    seq, mask = pad_protein_sequence(tcra_fv, max_seq_len)

    one_hot_feat = one_hot_encoding(seq)
    blosum50_feat = bolossum_encoding(seq, BLOSUM50_MATRIX)
    atchley_factor_feat = atchley_factor_encoding(seq)

    concat_feat = np.concatenate(
        (one_hot_feat, blosum50_feat, atchley_factor_feat), axis=1)
    return concat_feat, mask

def tcrb_fv_encoding(tcrb_fv, max_seq_len):
    # max_seq_len = ? for tcrb_fv
    seq, mask = pad_protein_sequence(tcrb_fv, max_seq_len)

    one_hot_feat = one_hot_encoding(seq)
    blosum50_feat = bolossum_encoding(seq, BLOSUM50_MATRIX)
    atchley_factor_feat = atchley_factor_encoding(seq)

    concat_feat = np.concatenate(
        (one_hot_feat, blosum50_feat, atchley_factor_feat), axis=1)
    return concat_feat, mask

def tcra_cdr3_encoding(tcra_cdr3, max_seq_len):
    # max_seq_len = ? for tcra_cdr3
    seq, mask = pad_protein_sequence(tcra_cdr3, max_seq_len)

    one_hot_feat = one_hot_encoding(seq)
    blosum50_feat = bolossum_encoding(seq, BLOSUM50_MATRIX)
    atchley_factor_feat = atchley_factor_encoding(seq)

    concat_feat = np.concatenate(
        (one_hot_feat, blosum50_feat, atchley_factor_feat), axis=1)
    return concat_feat

def tcrb_cdr3_encoding(tcrb_cdr3, max_seq_len):
    # max_seq_len = ? for tcrb_cdr3
    seq, mask = pad_protein_sequence(tcrb_cdr3, max_seq_len)

    one_hot_feat = one_hot_encoding(seq)
    blosum50_feat = bolossum_encoding(seq, BLOSUM50_MATRIX)
    atchley_factor_feat = atchley_factor_encoding(seq)

    concat_feat = np.concatenate(
        (one_hot_feat, blosum50_feat, atchley_factor_feat), axis=1)
    return concat_feat

if __name__ == '__main__':
    seq = 'AAAWYR'
    encoded_seq = one_hot_encoding(seq)
    print(encoded_seq)
    encoded_seq = bolossum_encoding(seq, BLOSUM50_MATRIX)
    print(encoded_seq)
    encoded_seq = atchley_factor_encoding(seq)
    print(encoded_seq)

    mhc = mhc_encoding(seq, 15)
    print(mhc.shape)