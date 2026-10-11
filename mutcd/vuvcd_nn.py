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
from pathlib import Path

AA_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def load_fasta(filepath):
    seq = ''
    with open(filepath) as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip()
    return seq.upper()

def nn_score_all(seq, window=17, weights=None):
    """Calculate NNHK helix/sheet scores for all residues"""
    if isinstance(weights, dict) and weights.get('kind') == 'hidden2':
        W1, b1, W2, b2 = (weights[k] for k in ('W1', 'b1', 'W2', 'b2'))
        if window != 17:
            raise ValueError('hidden2 model requires a 17-residue window')
        n = len(seq)
        X = np.zeros((n, 357), dtype=float)
        for pos in range(n):
            for col, offset in enumerate(range(-8, 9)):
                i = pos + offset
                aa = seq[i] if 0 <= i < n else '-'
                if aa not in AA_IDX:
                    raise ValueError(f'Unknown amino acid {aa!r} at position {i + 1}')
                X[pos, col * 21 + AA_IDX[aa]] = 1.0
        hidden = sigmoid(X @ W1 + b1)
        outputs = sigmoid(hidden @ W2 + b2)
        return outputs[:, 0], outputs[:, 1]
    wh, ws, bh, bs = weights if weights is not None else (W_HELIX, W_SHEET, BIAS_HELIX, BIAS_SHEET)
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
            xh += wh[idx, col]
            xs += ws[idx, col]
        helix_out[pos] = sigmoid(xh + bh)
        sheet_out[pos] = sigmoid(xs + bs)
    return helix_out, sheet_out

def load_model(name):
    if name == 'sp175_hidden2':
        root = Path(__file__).resolve().parent / 'models' / 'sp175' / 'hidden2'
        arrays = {
            key: np.load(root / filename, allow_pickle=False)
            for key, filename in {
                'W1': 'W1_hidden2_SP175.npy',
                'b1': 'b1_hidden2_SP175.npy',
                'W2': 'W2_hidden2_SP175.npy',
                'b2': 'b2_hidden2_SP175.npy',
            }.items()
        }
        expected = {'W1': (357, 2), 'b1': (2,), 'W2': (2, 2), 'b2': (2,)}
        for key, shape in expected.items():
            if arrays[key].shape != shape:
                raise ValueError(f'{key} must have shape {shape}, got {arrays[key].shape}')
        return {'kind': 'hidden2', **arrays}
    if name == 'hk1989':
        return W_HELIX, W_SHEET, BIAS_HELIX, BIAS_SHEET
    root = Path(__file__).resolve().parent / 'models' / 'sp175'
    wh = np.load(root / 'W_HELIX_SP175.npy', allow_pickle=False)
    ws = np.load(root / 'W_SHEET_SP175.npy', allow_pickle=False)
    bh = float(np.load(root / 'BIAS_HELIX_SP175.npy', allow_pickle=False))
    bs = float(np.load(root / 'BIAS_SHEET_SP175.npy', allow_pickle=False))
    if wh.shape != (21, 17) or ws.shape != (21, 17):
        raise ValueError('SP175 weights must have shape (21, 17)')
    return wh, ws, bh, bs


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
    parser.add_argument('--model', choices=['hk1989', 'sp175', 'sp175_hidden2'], default='hk1989')
    args = parser.parse_args()

    seq = load_fasta(args.fasta)
    n = len(seq)
    print(f'Sequence length: {n} residues')

    n_helix  = int(round(args.helix_frac  * n))
    n_strand = int(round(args.sheet_frac  * n))
    print(f'Target: Helix={n_helix} residues, Strand={n_strand} residues')

    helix_out, sheet_out = nn_score_all(seq, weights=load_model(args.model))
    state = assign_vuvcd_nn(seq, helix_out, sheet_out, n_helix, n_strand)
    print_map(seq, state, label=args.label, width=args.width)

if __name__ == '__main__':
    main()
