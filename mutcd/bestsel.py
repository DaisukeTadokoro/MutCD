"""
MutCD: bestsel.py
BeStSel-inspired CD spectral analysis with 8-component secondary structure
Based on Micsonai et al. (2015) PNAS and Matsuo & Gekko (2008) Proteins

Usage:
    python3 bestsel.py --exp sample_25C.csv sample_80C.csv
"""
import numpy as np
from scipy.optimize import nnls
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import csv
import re
import os
import sys

# ============================================================
# Data loading
# ============================================================

def load_sp175(data_dir):
    """Load SP175 reference dataset"""
    A = np.loadtxt(os.path.join(data_dir, 'A.txt'))   # (66, 71)
    wl = np.arange(240, 240-66, -1)[::-1]             # 175-240 ascending
    return A[::-1, :], wl                              # (66, 71) ascending


def load_ss8(ss_file, csv_file, lbl2_file):
    """Load 8-component SS fractions from bestsel_twist.py output"""
    # PDB -> SS8
    pdb_to_ss8 = {}
    with open(ss_file) as f:
        content = f.read()
    summary = re.search(r'=== サマリー ===\n(.+)', content, re.DOTALL)
    if summary:
        for line in summary.group(1).strip().split('\n')[1:]:
            parts = line.split()
            if len(parts) >= 9:
                pdb_to_ss8[parts[0].lower()] = [float(x) for x in parts[1:9]]

    # SP175.csv -> ID to PDB mapping
    id_to_pdb = {}
    with open(csv_file) as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4:
                pdb = row[2].strip().lower()
                sid = row[3].strip()
                if sid and pdb:
                    id_to_pdb[sid] = pdb

    # lbl2.txt order
    lbl2 = open(lbl2_file).read().split()

    SS8 = np.zeros((len(lbl2), 8))
    valid_idx = []
    for i, sid in enumerate(lbl2):
        pdb = id_to_pdb.get(sid, '')
        if pdb in pdb_to_ss8:
            SS8[i] = pdb_to_ss8[pdb]
            valid_idx.append(i)

    return SS8[valid_idx], valid_idx


def load_spectrum(filepath, wl_min=185, wl_max=240):
    """Load experimental CD spectrum"""
    wl, cd = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().replace(',', ' ').split()
            try:
                w, v = float(parts[0]), float(parts[1])
                if wl_min <= w <= wl_max:
                    wl.append(w); cd.append(v)
            except:
                continue
    wl = np.array(wl); cd = np.array(cd)
    order = np.argsort(wl)
    return wl[order], cd[order]


# ============================================================
# Wavelength optimization (Micsonai et al. 2015)
# ============================================================

def cv_rmsd_for_component(comp_i, A_use, SS8):
    """Leave-one-out CV RMSD for component i"""
    n_prot = A_use.shape[1]
    errors = []
    for j in range(n_prot):
        idx = [k for k in range(n_prot) if k != j]
        R = A_use[:, idx]
        cd_test = A_use[:, j]
        coeffs, _ = nnls(R, cd_test)
        total = coeffs.sum()
        if total > 0: coeffs /= total
        pred_i = np.dot(coeffs, SS8[idx, comp_i])
        errors.append((pred_i - SS8[j, comp_i])**2)
    return np.sqrt(np.mean(errors))


def find_optimal_wl(comp_i, A_asc, SS8, wl_grid,
                    wl_min=185, wl_max=240, window=10, step=5):
    """Find optimal wavelength range for component i"""
    full = (wl_grid >= wl_min) & (wl_grid <= wl_max)
    A_full = A_asc[full, :]
    baseline = cv_rmsd_for_component(comp_i, A_full, SS8)

    best_rmsd = baseline
    best_mask = full.copy()
    best_excl = None

    wl_s = wl_min
    while wl_s + window <= wl_max:
        excl = (wl_grid >= wl_s) & (wl_grid < wl_s + window)
        use = full & ~excl
        if use.sum() < 10:
            wl_s += step; continue
        rmsd = cv_rmsd_for_component(comp_i, A_asc[use, :], SS8)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_mask = use.copy()
            best_excl = (wl_s, wl_s + window)
        wl_s += step

    return best_mask, best_excl, baseline, best_rmsd


# ============================================================
# Fitting
# ============================================================

COMP8 = ['Helix1', 'Helix2', 'Anti1', 'Anti2', 'Anti3',
         'Parallel', 'Turn', 'Others']


def fit_bestsel(exp_wl, exp_cd, SS8, A_valid, wl_grid,
                optimal_masks, wl_min=185, wl_max=240):
    """Fit experimental spectrum using BeStSel-inspired method"""
    ss = np.zeros(8)
    for i in range(8):
        mask = optimal_masks[i]
        wl_use = wl_grid[mask]
        A_use = A_valid[mask, :]

        exp_mask = (exp_wl >= wl_min) & (exp_wl <= wl_max)
        f_exp = interp1d(exp_wl[exp_mask], exp_cd[exp_mask],
                        kind='cubic', bounds_error=False,
                        fill_value='extrapolate')
        cd_exp = f_exp(wl_use)

        coeffs, _ = nnls(A_use, cd_exp)
        total = coeffs.sum()
        if total > 0: coeffs /= total
        ss[i] = np.dot(coeffs, SS8[:, i])

    ss /= ss.sum()

    # RMSD
    full_mask = (wl_grid >= wl_min) & (wl_grid <= wl_max)
    wl_f = wl_grid[full_mask]
    A_f = A_valid[full_mask, :]
    exp_mask = (exp_wl >= wl_min) & (exp_wl <= wl_max)
    f_exp = interp1d(exp_wl[exp_mask], exp_cd[exp_mask],
                    kind='cubic', bounds_error=False,
                    fill_value='extrapolate')
    cd_f = f_exp(wl_f)
    coeffs_f, _ = nnls(A_f, cd_f)
    cd_fit = A_f @ coeffs_f
    rmsd = np.sqrt(np.mean((cd_f - cd_fit)**2))

    return ss, rmsd, wl_f, cd_f, cd_fit


# ============================================================
# Plotting
# ============================================================

def plot_results(results, output='bestsel_result.png'):
    n = len(results)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))

    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    comp_colors = ['#E53935', '#FF7043', '#1565C0',
                   '#1E88E5', '#42A5F5', '#2E7D32', '#FB8C00', '#7B1FA2']

    # CD spectrum + fit
    ax = axes[0]
    for i, (label, ss, rmsd, wl_f, cd_f, cd_fit) in enumerate(results):
        c = colors[i % len(colors)]
        ax.plot(wl_f, cd_f, color=c, alpha=0.3, linewidth=1,
                label=f'{label} obs')
        ax.plot(wl_f, cd_fit, color=c, linewidth=2,
                linestyle='--', label=f'{label} fit')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('CD (mdeg)')
    ax.set_title('CD Spectrum & Fit')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Bar charts
    for pi, (label, ss, rmsd, *_) in enumerate(results):
        ax = axes[pi + 1]
        vals = [v * 100 for v in ss]
        bars = ax.bar(COMP8, vals, color=comp_colors, edgecolor='white')
        for bar, v in zip(bars, vals):
            if v > 1:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f'{v:.1f}%', ha='center', va='bottom',
                        fontsize=8, fontweight='bold')
        ax.set_ylim(0, 45)
        ax.set_ylabel('Content (%)')
        h = (ss[0] + ss[1]) * 100
        a = ss[2:5].sum() * 100
        p = ss[5] * 100
        ax.set_title(f'{label}  RMSD={rmsd:.3f}\n'
                     f'Helix={h:.1f}% Anti={a:.1f}% Para={p:.1f}%',
                     fontsize=10)
        ax.tick_params(axis='x', rotation=35)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches='tight')
    print(f'[saved] {output}')
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='MutCD: BeStSel-inspired 8-component CD analysis')
    parser.add_argument('--exp', nargs='+', required=True,
                        help='Experimental CD spectrum files (.csv)')
    parser.add_argument('--labels', nargs='*',
                        help='Labels for each spectrum (default: filename)')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(
                            os.path.abspath(__file__)), '..', 'data', 'SP175'),
                        help='SP175 data directory')
    parser.add_argument('--ss_file', type=str, required=True,
                        help='bestsel_twist.py output file (SS8 summary)')
    parser.add_argument('--wlmin', type=float, default=185)
    parser.add_argument('--wlmax', type=float, default=240)
    parser.add_argument('--window', type=int, default=10,
                        help='Wavelength exclusion window size (nm)')
    parser.add_argument('--step', type=int, default=5,
                        help='Wavelength optimization step size (nm)')
    parser.add_argument('--out', type=str, default='bestsel_result.png',
                        help='Output figure filename')
    args = parser.parse_args()

    data_dir = args.data_dir
    csv_file  = os.path.join(data_dir, 'SP175.csv')
    lbl2_file = os.path.join(data_dir, 'lbl2.txt')

    # Load data
    print('Loading SP175 reference dataset...')
    A_asc, wl_grid = load_sp175(data_dir)
    SS8, valid_idx = load_ss8(args.ss_file, csv_file, lbl2_file)
    A_valid = A_asc[:, valid_idx]
    print(f'  {len(valid_idx)} proteins loaded')

    # Wavelength optimization
    print('Optimizing wavelength ranges...')
    optimal_masks = []
    for i in range(8):
        mask, excl, base, best = find_optimal_wl(
            i, A_valid, SS8, wl_grid,
            args.wlmin, args.wlmax, args.window, args.step)
        optimal_masks.append(mask)
        excl_str = f'{excl[0]}-{excl[1]}nm excluded' if excl else 'no exclusion'
        print(f'  {COMP8[i]:12s}: {excl_str} '
              f'(CV RMSD: {base:.4f} -> {best:.4f})')

    # Fit experimental spectra
    labels = args.labels or [os.path.basename(f) for f in args.exp]
    results = []
    print('\n=== Secondary structure analysis ===')
    for exp_file, label in zip(args.exp, labels):
        wl, cd = load_spectrum(exp_file, args.wlmin, args.wlmax)
        ss, rmsd, wl_f, cd_f, cd_fit = fit_bestsel(
            wl, cd, SS8, A_valid, wl_grid, optimal_masks,
            args.wlmin, args.wlmax)

        print(f'\n--- {label} ---')
        for c, v in zip(COMP8, ss):
            print(f'  {c:12s}: {v:.1%}')
        print(f'  Helix={ss[:2].sum():.1%}  '
              f'Anti={ss[2:5].sum():.1%}  '
              f'Para={ss[5]:.1%}  '
              f'RMSD={rmsd:.4f}')
        results.append((label, ss, rmsd, wl_f, cd_f, cd_fit))

    plot_results(results, output=args.out)


if __name__ == '__main__':
    main()
