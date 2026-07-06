"""
BeStSel Phase 1:
PDBファイルからβシートのtwist角を計算し、8成分SS分率を算出する

参考:
- Ho & Curmi (2002) J. Mol. Biol. 317:291-308
- Micsonai et al. (2015) PNAS 112:E3095
- BeStSelのtwist境界: +3°(Anti1/2境界), +23°(Anti2/3境界)

DSSPのbp符号: 正→parallel, 負→antiparallel

8成分:
  Helix1   (G: 3-10ヘリックス)
  Helix2   (H: αヘリックス)
  Anti1    (逆平行β, twist < +3°)
  Anti2    (逆平行β, +3° ≤ twist < +23°)
  Anti3    (逆平行β, twist ≥ +23°)
  Parallel (平行β)
  Turn     (T)
  Others   (残り全部)
"""

import numpy as np
from Bio.PDB import PDBParser, DSSP
import sys
import warnings
warnings.filterwarnings('ignore')


def run_dssp(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]
    dssp = DSSP(model, pdb_file, dssp='mkdssp')
    return model, dssp


def get_ca_coords(model):
    """全残基のCα座標を {(chain_id, resseq): array} で返す"""
    ca = {}
    for chain in model:
        for res in chain:
            if 'CA' in res:
                resseq = res.get_id()[1]
                ca[(chain.id, resseq)] = res['CA'].get_vector().get_array()
    return ca


def calc_twist_angle(ca1, ca2, ca3, ca4):
    """
    4つのCα座標からtwist角を計算 (Ho & Curmi 2002)
    ca1→ca2: strand Aの方向ベクトル
    ca3→ca4: strand Bの方向ベクトル
    """
    v1 = ca2 - ca1
    v2 = ca4 - ca3

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None

    v1 = v1 / n1
    v2 = v2 / n2

    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_theta))

    # 符号: 外積とストランド間ベクトルの内積で右巻き/左巻きを判定
    cross = np.cross(v1, v2)
    inter = ca3 - ca1
    norm_inter = np.linalg.norm(inter)
    if norm_inter > 1e-6:
        inter = inter / norm_inter
        if np.dot(cross, inter) < 0:
            theta = -theta

    return theta


def calc_8component_ss(pdb_file, verbose=True):
    """
    PDBファイルから8成分SS分率を計算
    """
    model, dssp = run_dssp(pdb_file)
    ca_coords = get_ca_coords(model)
    keys = list(dssp.keys())
    total = len(keys)

    # resseq → key のマップ（bp番号から残基を探すため）
    resseq_to_key = {}
    for key in keys:
        resseq = key[1][1]
        chain = key[0]
        resseq_to_key[(chain, resseq)] = key

    counts = {
        'Helix1': 0, 'Helix2': 0,
        'Anti1': 0, 'Anti2': 0, 'Anti3': 0,
        'Parallel': 0, 'Turn': 0, 'Others': 0,
    }

    for key in keys:
        data = dssp[key]
        ss   = data[2]   # SS記号
        bp1  = data[6]   # beta partner 1 (符号付き残基番号)
        bp2  = data[8]   # beta partner 2 (符号付き残基番号)
        # ※ BioPythonのDSSP: index6=bp1_resnum, index7=bp1_energy,
        #                      index8=bp2_resnum, index9=bp2_energy

        if ss == 'G':
            counts['Helix1'] += 1
        elif ss == 'H':
            counts['Helix2'] += 1
        elif ss == 'T':
            counts['Turn'] += 1
        elif ss == 'E':
            # bp符号で parallel/antiparallel を判定
            # 正→parallel, 負→antiparallel
            is_parallel = False
            bp_resnum = None

            for bp in [bp1, bp2]:
                if bp == 0:
                    continue
                if bp > 0:
                    is_parallel = True
                    bp_resnum = bp
                elif bp < 0:
                    is_parallel = False
                    bp_resnum = -bp
                break  # 最初のbpを使用

            if is_parallel:
                counts['Parallel'] += 1
            else:
                # twist角を計算してAnti1/2/3に分類
                twist = None
                chain_id = key[0]
                res_i = key[1][1]

                # 隣の残基 (i+1)
                ca_i = ca_coords.get((chain_id, res_i))
                ca_i1 = ca_coords.get((chain_id, res_i + 1))

                if bp_resnum is not None:
                    ca_j = ca_coords.get((chain_id, bp_resnum))
                    ca_j1 = ca_coords.get((chain_id, bp_resnum + 1))

                    if all(x is not None for x in [ca_i, ca_i1, ca_j, ca_j1]):
                        twist = calc_twist_angle(
                            np.array(ca_i),  np.array(ca_i1),
                            np.array(ca_j),  np.array(ca_j1)
                        )

                if twist is None:
                    counts['Anti2'] += 1  # デフォルト
                elif twist < 3.0:
                    counts['Anti1'] += 1
                elif twist < 23.0:
                    counts['Anti2'] += 1
                else:
                    counts['Anti3'] += 1
        else:
            counts['Others'] += 1

    fractions = {k: v / total for k, v in counts.items()}

    if verbose:
        name = pdb_file.split('/')[-1]
        print(f"\n=== {name} (総残基数: {total}) ===")
        for k, v in counts.items():
            print(f"  {k:10s}: {v:4d} ({fractions[k]:.3f})")

    return fractions, counts, total


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("使い方: python3 bestsel_twist.py <pdb_file> [pdb_file2 ...]")
        sys.exit(1)

    all_results = {}
    components = ['Helix1', 'Helix2', 'Anti1', 'Anti2', 'Anti3', 'Parallel', 'Turn', 'Others']

    for pdb_file in sys.argv[1:]:
        try:
            fractions, counts, total = calc_8component_ss(pdb_file)
            all_results[pdb_file] = fractions
        except Exception as e:
            print(f"[エラー] {pdb_file}: {e}")
            import traceback; traceback.print_exc()

    if len(all_results) > 1:
        print("\n=== サマリー ===")
        print(f"{'PDB':15s} " + " ".join(f"{c:9s}" for c in components))
        for pdb, fracs in all_results.items():
            name = pdb.split('/')[-1].replace('.pdb', '')
            vals = " ".join(f"{fracs[c]:.3f}    " for c in components)
            print(f"{name:15s} {vals}")
