"""
MutCD: mutation_scan.py
β-strand stabilizing mutation scan using Holley-Karplus (1989) NNHK weights

Usage:
    python3 mutation_scan.py --fasta input.fasta --target_start 142 --target_end 151
"""
import numpy as np
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nnhk_weights import W_HELIX, W_SHEET, BIAS_HELIX, BIAS_SHEET, AA_ORDER

AA_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}
AA_STANDARD = list('ACDEFGHIKLMNPQRSTVWY')

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def nn_score(seq, pos, window=17):
    half = window // 2
    n = len(seq)
    xh = xs = 0.0
    for offset in range(-half, half + 1):
        col = offset + half
        i = pos + offset
        idx = AA_IDX.get(seq[i] if 0 <= i < n else '-', AA_IDX['-'])
        xh += W_HELIX[idx, col]
        xs += W_SHEET[idx, col]
    return sigmoid(xh + BIAS_HELIX), sigmoid(xs + BIAS_SHEET)

def region_score(seq, start, end):
    sh, he = [], []
    for i in range(start, end):
        h, s = nn_score(seq, i)
        he.append(h); sh.append(s)
    return np.mean(sh), np.mean(he)

def load_fasta(filepath):
    seq = ''
    with open(filepath) as f:
        for line in f:
            if not line.startswith('>'):
                seq += line.strip()
    return list(seq.rstrip("H"))

def mutation_scan(seq, target_start, target_end, context_margin=6, exclude=None):
    """
    Scan mutations in target region
    target_start, target_end: 1-indexed, inclusive
    exclude: list of 1-indexed positions to exclude (e.g. catalytic residues)
    """
    ts = target_start - 1  # 0-indexed
    te = target_end        # 0-indexed exclusive
    cs = max(0, ts - context_margin)
    ce = min(len(seq), te + context_margin)
    exclude = [e - 1 for e in (exclude or [])]  # 0-indexed

    wt_sheet, wt_helix = region_score(seq, cs, ce)

    results = []
    for pos in range(ts, te):
        if pos in exclude:
            continue
        wt_aa = seq[pos]
        for mut_aa in AA_STANDARD:
            if mut_aa == wt_aa:
                continue
            mut_seq = seq.copy()
            mut_seq[pos] = mut_aa
            mt_s, mt_h = region_score(mut_seq, cs, ce)
            ds = mt_s - wt_sheet
            dh = mt_h - wt_helix
            results.append({
                'pos': pos + 1,
                'wt': wt_aa,
                'mut': mut_aa,
                'mutation': f'{wt_aa}{pos+1}{mut_aa}',
                'delta_sheet': ds,
                'delta_helix': dh,
                'score': ds - dh,
                'sheet_after': mt_s,
            })

    results.sort(key=lambda x: -x['score'])
    return results, wt_sheet, wt_helix

def main():
    parser = argparse.ArgumentParser(
        description='MutCD: β-strand stabilizing mutation scan')
    parser.add_argument('--fasta', required=True,
                        help='Input FASTA file')
    parser.add_argument('--target_start', type=int, required=True,
                        help='Target region start (1-indexed)')
    parser.add_argument('--target_end', type=int, required=True,
                        help='Target region end (1-indexed, inclusive)')
    parser.add_argument('--exclude', type=int, nargs='*', default=[],
                        help='Positions to exclude (1-indexed, e.g. catalytic residues)')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top mutations to show (default: 20)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output CSV file (optional)')
    args = parser.parse_args()

    seq = load_fasta(args.fasta)
    print(f'配列長: {len(seq)}残基')
    print(f'対象領域: {args.target_start}-{args.target_end}番目')
    print(f'野生型配列: {"".join(seq[args.target_start-1:args.target_end])}')
    if args.exclude:
        print(f'除外位置: {args.exclude}')

    results, wt_sheet, wt_helix = mutation_scan(
        seq, args.target_start, args.target_end, exclude=args.exclude)

    print(f'\n野生型 Sheet={wt_sheet:.4f}, Helix={wt_helix:.4f}')
    print(f'\n=== Top {args.top} β-strand stabilizing mutations ===')
    print(f'{"Mutation":8s} {"ΔSheet":8s} {"ΔHelix":8s} {"Score":8s} {"Sheet_after":10s}')
    print('-' * 52)
    for r in results[:args.top]:
        print(f'{r["mutation"]:8s} {r["delta_sheet"]:+.4f}   '
              f'{r["delta_helix"]:+.4f}   {r["score"]:+.4f}   {r["sheet_after"]:.4f}')

    print('\n=== Best mutation per position ===')
    best = {}
    for r in results:
        p = r['pos']
        if p not in best or r['score'] > best[p]['score']:
            best[p] = r
    for p in sorted(best):
        r = best[p]
        print(f'  {r["mutation"]:8s}: ΔSheet={r["delta_sheet"]:+.4f}, '
              f'ΔHelix={r["delta_helix"]:+.4f}, Score={r["score"]:+.4f}')

    if args.out:
        import pandas as pd
        pd.DataFrame(results).to_csv(args.out, index=False)
        print(f'\n結果を保存: {args.out}')

if __name__ == '__main__':
    main()
