"""
MutCD: vuvcd_nn.py
VUVCD-NN method for residue-level secondary structure mapping
Based on Matsuo & Gekko (2008) Proteins and Holley-Karplus (1989) PNAS

Usage:
    python3 vuvcd_nn.py --fasta input.fasta --helix_frac 0.216 --sheet_frac 0.318
"""
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnhk_weights import W_HELIX, W_SHEET, BIAS_HELIX, BIAS_SHEET, AA_ORDER

AA_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_fasta(filepath):
    seq = ''
    with open(filepath) as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip()
    return seq.rstrip("H")

def nn_score_all(seq, window=17):
    """Calculate NNHK helix/sheet scores for all residues"""
    half = window // 2
    n = len(seq)
    helix_out = np.zeros(n)
    sheet_out = np.zeros(n)
    for pos in range(n):
        xh = xs = 0.0
        for offset in range(-half, half + 1):
            col = offset + half
            i = pos + offset
            idx = AA_IDX.get(seq[i] if 0 <= i < n else '-', AA_IDX['-'])
            xh += W_HELIX[idx, col]
            xs += W_SHEET[idx, col]
        helix_out[pos] = sigmoid(xh + BIAS_HELIX)
        sheet_out[pos] = sigmoid(xs + BIAS_SHEET)
    return helix_out, sheet_out

def filter_min_segment(state, label, min_len):
    state = state.copy()
    n = len(state)
    i = 0
    while i < n:
        if state[i] == label:
            j = i
            while j < n and state[j] == label:
                j += 1
            if (j - i) < min_len:
                state[i:j] = 'O'
            i = j
        else:
            i += 1
    return state

def assign_vuvcd_nn(seq, helix_out, sheet_out,
                    n_helix_target, n_strand_target,
                    min_helix_len=4, min_strand_len=2):
    """
    VUVCD-NN Step 2-5 (Matsuo & Gekko 2008):
    Assign H/E/O states using NNHK scores as ranking,
    constrained by CD-derived residue counts
    """
    n = len(seq)
    state = np.full(n, 'O', dtype='<U1')

    for idx in np.argsort(-helix_out):
        if np.sum(state == 'H') >= n_helix_target:
            break
        if state[idx] == 'O':
            state[idx] = 'H'

    for idx in np.argsort(-sheet_out):
        if np.sum(state == 'E') >= n_strand_target:
            break
        if state[idx] == 'O':
            state[idx] = 'E'

    state = filter_min_segment(state, 'H', min_helix_len)
    state = filter_min_segment(state, 'E', min_strand_len)
    return state

def print_map(seq, state, label='', width=60):
    print(f'\n=== {label} ===')
    n = len(seq)
    for start in range(0, n, width):
        end = min(start + width, n)
        print(f'{start+1:4d} {seq[start:end]}')
        print(f'     {"".join(state[start:end])}')
    n_h = np.sum(state == 'H')
    n_e = np.sum(state == 'E')
    print(f'Helix: {n_h} ({n_h/n:.1%})  '
          f'Strand: {n_e} ({n_e/n:.1%})  '
          f'Other: {n-n_h-n_e} ({(n-n_h-n_e)/n:.1%})')

def main():
    parser = argparse.ArgumentParser(
        description='MutCD: VUVCD-NN residue-level secondary structure mapping')
    parser.add_argument('--fasta', required=True,
                        help='Input FASTA file')
    parser.add_argument('--helix_frac', type=float, required=True,
                        help='Helix fraction from CD analysis (e.g. 0.216)')
    parser.add_argument('--sheet_frac', type=float, required=True,
                        help='Sheet fraction from CD analysis (e.g. 0.318)')
    parser.add_argument('--label', type=str, default='Sample',
                        help='Sample label (e.g. 25C)')
    parser.add_argument('--width', type=int, default=60,
                        help='Line width for sequence map display')
    args = parser.parse_args()

    seq = load_fasta(args.fasta)
    n = len(seq)
    print(f'Sequence length: {n} residues')

    n_helix  = int(round(args.helix_frac  * n))
    n_strand = int(round(args.sheet_frac  * n))
    print(f'Target: Helix={n_helix} residues, Strand={n_strand} residues')

    helix_out, sheet_out = nn_score_all(seq)
    state = assign_vuvcd_nn(seq, helix_out, sheet_out, n_helix, n_strand)
    print_map(seq, state, label=args.label, width=args.width)

if __name__ == '__main__':
    main()
