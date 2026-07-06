# MutCD

**Mutation design guided by circular dichroism (CD) spectroscopy**

MutCD is a computational pipeline that integrates experimental CD spectroscopy with sequence-based neural network analysis to identify thermally labile regions in proteins and predict stabilizing mutations.

---

## Background

Conventional protein engineering tools (AlphaFold, FoldX, etc.) predict structure or stability from static snapshots. MutCD takes a different approach: it uses **CD spectroscopy** — which captures dynamic conformational changes in solution — as the starting point for mutation design.

The key insight is that CD spectroscopy can reveal structural changes (e.g., parallel β-sheet loss upon thermal denaturation) that are invisible to X-ray crystallography or structure prediction tools. By identifying *which secondary structure elements change* and *where in the sequence they are located*, MutCD provides mechanistically grounded mutation candidates.

The pipeline integrates:
- **BeStSel-inspired analysis** (Micsonai et al., 2015 PNAS) for 8-component secondary structure decomposition
- **VUVCD-NN method** (Matsuo & Gekko, 2008 Proteins) for residue-level structural mapping
- **Holley-Karplus neural network** (Holley & Karplus, 1989 PNAS) for sequence-based β-sheet scoring
- **SP175 reference dataset** (Lees et al., 2006 Bioinformatics) via DichroWebGit

---

## Installation

```bash
git clone https://github.com/xxx/MutCD.git
cd MutCD
pip install -r requirements.txt
```

---

## Pipeline Overview

```
Experimental CD spectra (25°C / denaturation temperature)
        ↓
1. bestsel.py       → 8-component SS analysis (Helix1/2, Anti1/2/3, Parallel, Turn, Others)
        ↓
2. vuvcd_nn.py      → Residue-level mapping of structural changes
        ↓
3. mutation_scan.py → β-sheet stabilizing mutation scan (NNHK Sheet score)
```

---

## Usage

### Step 1: CD spectral analysis (8-component)

```bash
python3 mutcd/bestsel.py \
  --exp sample_25C.csv sample_80C.csv \
  --labels 25C 80C \
  --ss_file data/SP175/sp175_8comp_ss_final.txt \
  --out result.png
```

**Output:**
- 8-component secondary structure fractions (Helix1/2, Anti1/2/3, Parallel, Turn, Others)
- RMSD of spectral fit
- Figure: CD spectra + bar charts

---

### Step 2: Residue-level structural mapping

```bash
python3 mutcd/vuvcd_nn.py \
  --fasta protein.fasta \
  --helix_frac 0.216 \
  --sheet_frac 0.318 \
  --label 25C
```

Run for both temperatures and compare outputs to identify thermally labile regions.

**Output:**
- H/E/O assignment for each residue
- Sequence map

---

### Step 3: Mutation scan

```bash
python3 mutcd/mutation_scan.py \
  --fasta protein.fasta \
  --target_start 142 \
  --target_end 151 \
  --exclude 152 153 154 155 156 \
  --top 20 \
  --out mutations.csv
```

**Output:**
- Ranked list of β-sheet stabilizing mutations (ΔSheet, ΔHelix, Score)
- Best mutation per position

---

## Example: FAST PETase

```bash
# Step 1: CD analysis
python3 mutcd/bestsel.py \
  --exp examples/fast_petase/EV1_el3taka_agg_25deg_100um.csv \
        examples/fast_petase/EV1_el3taka_agg_80deg_100um.csv \
  --labels 25C 80C \
  --ss_file data/SP175/sp175_8comp_ss_final.txt \
  --out examples/fast_petase/bestsel_result.png

# Step 2: Residue mapping
python3 mutcd/vuvcd_nn.py \
  --fasta examples/fast_petase/rcsb_pdb_7SH6.fasta \
  --helix_frac 0.216 --sheet_frac 0.318 --label 25C

python3 mutcd/vuvcd_nn.py \
  --fasta examples/fast_petase/rcsb_pdb_7SH6.fasta \
  --helix_frac 0.202 --sheet_frac 0.282 --label 80C

# Step 3: Mutation scan (catalytic site excluded)
python3 mutcd/mutation_scan.py \
  --fasta examples/fast_petase/rcsb_pdb_7SH6.fasta \
  --target_start 142 --target_end 151 \
  --exclude 152 153 154 155 156 \
  --top 20
```

**Result (FAST PETase):**

| Mutation | ΔSheet | ΔHelix | Score | ThermoProt ΔP | DynaMut2 ΔΔG |
|----------|--------|--------|-------|---------------|---------------|
| K142T | +0.081 | -0.015 | +0.096 | +0.039 | -1.22 kcal/mol |
| D144T | +0.095 | -0.008 | +0.103 | +0.046 | -0.64 kcal/mol |
| A146T | +0.081 | -0.024 | +0.105 | +0.009 | +0.16 kcal/mol |
| R147T | +0.082 | -0.018 | +0.099 | +0.000 | -0.79 kcal/mol |

**K142T and D144T are prioritized as top candidates** (consistent across all three methods).

---

## SP175 Reference Dataset

The `data/SP175/` directory contains:

| File | Description |
|------|-------------|
| `A.txt` | CD spectra of 71 proteins (DichroWebGit) |
| `F.txt` | 4-component SS fractions (DichroWebGit) |
| `lbl2.txt` | Protein ID list in A.txt/F.txt order (DichroWebGit) |
| `SP175.csv` | Validated 68-protein PDB correspondence table (MutCD) |
| `sp175_8comp_ss_final.txt` | 8-component SS fractions from PDB twist angle calculation (MutCD) |

### Note on SP175.csv

`SP175.csv` was constructed by cross-referencing:
- Lees et al. (2006) Bioinformatics Table 1S
- DichroWebGit lbl2.txt
- RCSB PDB structures
- F.txt SS values for validation

Three proteins were excluded (BGAL/1bgl: DSSP error, HGDC: PDB unknown, PELC: duplicate).

### Recomputing sp175_8comp_ss_final.txt (optional)

If you wish to recompute from PDB files:

```bash
# Download PDB files listed in data/SP175/SP175.csv
# Place them in data/SP175/pdb/

python3 mutcd/bestsel_twist.py data/SP175/pdb/*.pdb \
  2>/dev/null > data/SP175/sp175_8comp_ss_final.txt
```

---

## Methods

### BeStSel-inspired analysis (bestsel.py)
Reimplemented from Micsonai et al. (2015) PNAS 112:E3095.
- 8 secondary structure components (Helix1/2, Anti1/2/3, Parallel, Turn, Others)
- Leave-one-out cross-validated wavelength range optimization
- Non-negative least squares (NNLS) fitting

### VUVCD-NN mapping (vuvcd_nn.py)
Based on Matsuo & Gekko (2008) Proteins 72:104.
- Holley-Karplus (1989) neural network weights for residue scoring
- CD-derived residue counts constrain the assignment

### Mutation scan (mutation_scan.py)
Novel application of Holley-Karplus (1989) PNAS 86:152 weights.
- Sheet score maximization across target region
- Score = ΔSheet − ΔHelix

---

## References

1. Micsonai et al. (2015) *PNAS* 112:E3095 — BeStSel
2. Holley & Karplus (1989) *PNAS* 86:152 — Neural network for SS prediction
3. Matsuo & Gekko (2008) *Proteins* 72:104 — VUVCD-NN method
4. Lees et al. (2006) *Bioinformatics* 22:1955 — SP175 dataset
5. Ho & Curmi (2002) *J. Mol. Biol.* 317:291 — β-sheet twist angle definition
6. Kabsch & Sander (1983) *Biopolymers* 22:2577 — DSSP

---

## Acknowledgments

Algorithm development and implementation were assisted by Claude (Anthropic).
The conceptual integration of CD spectroscopy with the Holley-Karplus neural network
for mutation design emerged through iterative human-AI collaboration.

SP175 reference data sourced from DichroWebGit (https://github.com/pcddb/DichroWebGit).

---

## License

MIT License
