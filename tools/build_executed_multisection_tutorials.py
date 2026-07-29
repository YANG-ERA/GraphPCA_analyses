"""Build compact, public tutorial assets from audited full-atlas results.

The script is intentionally local-only: it reads the complete research results
from the mounted project drives, writes deterministic down-sampled plot inputs
to ``source/_static``, and rewrites the two atlas-dependent notebooks.  The
published notebooks then execute without the private atlas files while keeping
their figures reproducible from the bundled compact inputs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad
import nbformat as nbf
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
STATIC = SOURCE / "_static"
RNG = np.random.default_rng(666)

MERFISH = Path("/Users/era/Desktop/ST_GraphPCA+/MERFISH_Aging_rotation_aligned_upload")
MERFISH_SECTIONS = Path("/Volumes/T7 Shield/ST_GPCA+/rebuttal/scratch/MERFISH_Aging/functional_validation_v2/prepared/sections")
MOSTA_STORE = Path("/Volumes/T7 Shield/ST_GPCA+/rebuttal/scratch/MOSTA_FAST26_GraphPCA_Turbo/02_section_stores/full")
MOSTA_MODEL = Path("/Volumes/T7 Shield/ST_GPCA+/rebuttal/results/hierarchical_multisample/MOSTA_FAST26/full_grouped_stage/eta_0p005")
MOSTA_INTERP = Path("/Volumes/T7 Shield/ST_GPCA+/rebuttal/results/hierarchical_multisample/MOSTA_FAST26/interpretability_grouped_stage_eta0p005_aligned_final")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def save_notebook(path: Path, cells):
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    notebook.metadata.language_info = {"name": "python", "version": "3"}
    nbf.write(notebook, path)


def build_merfish_assets():
    out = STATIC / "merfish_aging_tutorial"
    out.mkdir(parents=True, exist_ok=True)
    cell_index = pd.read_csv(MERFISH / "00_manifest/cell_index.csv.gz", dtype={"Unnamed: 0": str})
    cell_index = cell_index.set_index("Unnamed: 0")
    z = np.load(MERFISH / "05_W_interpretability/eta_0p02__rho_0p14872615/rotated_arrays/Z_all_varimax.npy", mmap_mode="r")
    chosen = ["MsBrainAgingSpatialDonor_1-0", "MsBrainAgingSpatialDonor_6-1", "MsBrainAgingSpatialDonor_12-1"]
    frames = []
    for name in chosen:
        one = ad.read_h5ad(MERFISH_SECTIONS / f"{name}.h5ad", backed="r")
        identifiers = one.obs_names.astype(str)
        positions = cell_index.reindex(identifiers)["analysis_position"]
        if positions.isna().any():
            raise RuntimeError(f"MERFISH result index does not cover {name}.")
        take = np.sort(RNG.choice(len(one), size=min(3000, len(one)), replace=False))
        xy = np.asarray(one.obsm["spatial"])[take, :2]
        obs = one.obs.iloc[take]
        frames.append(pd.DataFrame({
            "section": name,
            "age": obs["age"].astype(str).to_numpy(),
            "tissue": obs["tissue"].astype(str).to_numpy(),
            "clust_annot": obs["clust_annot"].astype(str).to_numpy(),
            "x": xy[:, 0], "y": xy[:, 1],
            "P08": np.asarray(z[positions.to_numpy(dtype=int)[take], 7], dtype=np.float32),
        }))
        one.file.close()
    scores = pd.concat(frames, ignore_index=True)
    scores.to_csv(out / "three_section_P08_scores.csv.gz", index=False, compression="gzip")

    genes = pd.read_csv(MERFISH / "00_manifest/retained_genes.csv")
    gene_col = "gene" if "gene" in genes.columns else genes.columns[0]
    w0 = np.load(MERFISH / "05_W_interpretability/eta_0p02__rho_0p14872615/rotated_arrays/W0_varimax.npy")
    pd.DataFrame({"gene": genes[gene_col].astype(str), "loading": w0[:, 7]}).nlargest(12, "loading").to_csv(out / "P08_top_loadings.csv", index=False)

    diagnostics = json.loads((MERFISH / "02_models/eta_0p02__rho_0p14872615/diagnostics.json").read_text())
    summary = {"n_cells": int(z.shape[0]), "n_sections": 31, "n_genes": int(w0.shape[0]), "n_components": int(w0.shape[1]),
               "converged": bool(diagnostics.get("converged", True)), "n_iter": int(diagnostics.get("n_iter", diagnostics.get("iterations", 51))),
               "source": "audited eta=0.02 shared-loading MERFISH fit"}
    (out / "model_summary.json").write_text(json.dumps(summary, indent=2))


def build_mosta_assets():
    out = STATIC / "mosta_fast26_tutorial"
    out.mkdir(parents=True, exist_ok=True)
    rotation = np.load(MOSTA_INTERP / "arrays/program_rotation.npy")
    selected = [
        ("E12.5_E1S1.MOSTA", "E12.5"),
        ("E14.5_E1S1.MOSTA", "E14.5"),
        ("E16.5_E1S1.MOSTA", "E16.5"),
    ]
    frames = []
    for section, stage in selected:
        metadata = pd.read_parquet(MOSTA_STORE / "metadata" / f"{section}_metadata.parquet")
        embedding = np.load(MOSTA_MODEL / "aligned_to_global_final/embeddings" / f"{section}_Z.npy", mmap_mode="r")
        take = np.sort(RNG.choice(len(metadata), size=min(4500, len(metadata)), replace=False))
        score = np.asarray(embedding[take]) @ rotation
        frames.append(pd.DataFrame({
            "section": section, "stage": stage,
            "annotation": metadata.iloc[take]["annotation"].astype(str).to_numpy(),
            "x": metadata.iloc[take]["spatial_x"].to_numpy(), "y": metadata.iloc[take]["spatial_y"].to_numpy(),
            "P02": score[:, 1], "P04": score[:, 3], "P05": score[:, 4],
        }))
    pd.concat(frames, ignore_index=True).to_csv(out / "three_stage_program_scores.csv.gz", index=False, compression="gzip")
    pd.read_csv(MOSTA_INTERP / "tables/program_top_genes.csv").query("program in ['P02', 'P04', 'P05'] and arm == 'positive' and rank <= 10").to_csv(out / "program_top_genes.csv", index=False)
    activity = pd.read_csv(MOSTA_INTERP / "tables/section_program_activity.csv")
    activity.query("program in ['P02', 'P04', 'P05']").to_csv(out / "section_program_activity.csv", index=False)
    arrays = np.load(MOSTA_INTERP / "arrays/stage_program_loadings.npz")
    stages = ["E12.5", "E13.5", "E14.5", "E15.5", "E16.5"]
    distance = np.array([[np.linalg.norm(arrays[a] - arrays[b], ord="fro") for b in stages] for a in stages])
    pd.DataFrame(distance, index=stages, columns=stages).rename_axis("stage").reset_index().to_csv(out / "aligned_stage_loading_distances.csv", index=False)
    summary = json.loads((MOSTA_MODEL / "model_summary.json").read_text())
    (out / "model_summary.json").write_text(json.dumps(summary, indent=2))


def write_merfish_notebook():
    cells = [
        md("""# Tutorial 5: MERFISH Aging multi-donor analysis\n\nThis executable tutorial visualizes compact, plot-ready inputs derived from the audited 31-section, 376,107-cell shared-loading fit. The full atlas is not distributed; the final section gives the local full-data fitting contract."""),
        md("""## Goal\n\nInspect a common rotated loading axis, map it in three real sections, and relate its scores to post hoc cell annotations. Program interpretation is descriptive: annotations were not used for fitting."""),
        code("""from pathlib import Path\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nsns.set_theme(style='white', context='notebook')\nplt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})\nASSET_DIR = next(path for path in [Path('_static/merfish_aging_tutorial'), Path('source/_static/merfish_aging_tutorial')] if path.exists())\nsummary = json.loads((ASSET_DIR / 'model_summary.json').read_text())\nscores = pd.read_csv(ASSET_DIR / 'three_section_P08_scores.csv.gz')\nsummary"""),
        md("""## 1. Full-data preprocessing and fitting contract\n\nFor a local rerun, split the normalized 374-gene MERFISH object by `slice_id`, take `obsm['spatial'][:, :2]` for each section, and fit the same 30-component hierarchical model. The published figures below use the converged saved result rather than silently re-fitting an atlas in the documentation build."""),
        code("""RUN_FULL_FIT = False\nif RUN_FULL_FIT:\n    import scanpy as sc\n    from GraphPCA import Run_Hierarchical_Multi_GPCA\n    adata = sc.read_h5ad('/path/to/MERFISH_Aging.h5ad')\n    sections = [adata[adata.obs['slice_id'].eq(name)].copy() for name in adata.obs['slice_id'].unique()]\n    locations = [one.obsm['spatial'][:, :2] for one in sections]\n    Zs, W0, Ws, info = Run_Hierarchical_Multi_GPCA(\n        sections, locations, n_components=30, lambdas=0.5, rhos=0.14872615,\n        sample_weights='equal_slice', max_iter=50, random_seed=666, return_info=True)\n    print(info.converged, info.n_iter)\nelse:\n    print('Full atlas fit is intentionally disabled in the published tutorial.')"""),
        md("""## 2. Map one common rotated program\n\n`P08` is the eighth coordinate after one common rotation of the shared loading matrix and all section embeddings. The common rotation makes the displayed coordinate comparable across sections."""),
        code("""sections = scores['section'].drop_duplicates().tolist()\nlo, hi = scores['P08'].quantile([0.01, 0.99])\nfig, axes = plt.subplots(1, len(sections), figsize=(12, 3.6), constrained_layout=True)\nfor ax, section in zip(axes, sections):\n    frame = scores.loc[scores.section.eq(section)]\n    points = ax.scatter(frame.x, frame.y, c=frame.P08, s=2, cmap='RdBu_r', vmin=lo, vmax=hi, linewidths=0, rasterized=True)\n    ax.set(title=f\"{section} ({frame.age.iloc[0]})\", aspect='equal'); ax.invert_yaxis(); ax.axis('off')\nfig.colorbar(points, ax=axes, shrink=0.75, label='rotated P08 score')\nplt.show()"""),
        md("""## 3. Inspect leading genes\n\nLoading signs and order are fixed only after the recorded common rotation. Positive leading genes provide molecular evidence for the displayed program; they do not by themselves establish a cell-state mechanism."""),
        code("""loadings = pd.read_csv(ASSET_DIR / 'P08_top_loadings.csv').sort_values('loading')\nplt.figure(figsize=(5.4, 4.4))\nplt.barh(loadings.gene, loadings.loading, color='#99000D')\nplt.xlabel('Rotated shared-loading weight'); plt.title('P08 leading positive genes')\nplt.show()\nloadings.sort_values('loading', ascending=False).head()"""),
        md("""## 4. Post hoc annotation summary\n\nThis section is a visualization aid only. It summarizes P08 within the observed annotations and makes no age, P24, causal, or absolute-activity claim."""),
        code("""annotation_summary = (scores.groupby('clust_annot', as_index=False)\n    .agg(mean_P08=('P08', 'mean'), n_cells=('P08', 'size'))\n    .query('n_cells >= 80').sort_values('mean_P08'))\nplot = annotation_summary.tail(12)\nplt.figure(figsize=(6, 4.8))\nplt.barh(plot.clust_annot, plot.mean_P08, color='#E34A33')\nplt.xlabel('Mean rotated P08 score in displayed sample'); plt.title('Post hoc annotation summary')\nplt.show()\nplot.sort_values('mean_P08', ascending=False)"""),
        md("""## Checks and caveats\n\nThe compact assets are deterministic samples from the converged, audited fit. They support visualization and teaching, not re-estimation of performance metrics. For donor-held-out projection, use `Project_Hierarchical_Multi_GPCA` with a loading centre fitted without the held-out donor; do not infer pseudotime from these data."""),
    ]
    save_notebook(SOURCE / "Tutorial5_MERFISH_Aging.ipynb", cells)


def write_mosta_notebook():
    cells = [
        md("""# Tutorial 7: MOSTA FAST26 multi-section analysis\n\nThis executable tutorial displays compact inputs derived from the converged 2.32-million-position, 26-section, stage-grouped fit. Known developmental stages are sample labels used for partial pooling; they are not fitted pseudotime."""),
        md("""## Goal\n\nReview the local preprocessing contract, display aligned molecular programs across three stages, inspect loading genes, and quantify loading-centre geometry without making a developmental trajectory claim."""),
        code("""from pathlib import Path\nimport json\nimport pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nsns.set_theme(style='white', context='notebook')\nplt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})\nASSET_DIR = next(path for path in [Path('_static/mosta_fast26_tutorial'), Path('source/_static/mosta_fast26_tutorial')] if path.exists())\nsummary = json.loads((ASSET_DIR / 'model_summary.json').read_text())\nscores = pd.read_csv(ASSET_DIR / 'three_stage_program_scores.csv.gz')\nsummary"""),
        md("""## 1. Local preprocessing and stage-aware fitting contract\n\nThe full workflow reads each FAST26 H5AD locally, uses raw `layers['count']`, performs library-size normalization and `log1p`, selects a stage-balanced 2,000-gene panel, constructs independent within-section graphs, and writes a disk-backed section store. The fit converged in 30 iterations on the complete atlas; it is not recomputed in this web tutorial."""),
        code("""RUN_FULL_ATLAS = False\nif RUN_FULL_ATLAS:\n    from pathlib import Path\n    from GraphPCA import Run_Hierarchical_Multi_GPCA\n    store_dir = Path('/path/to/MOSTA_section_store')\n    output_dir = Path('/path/to/MOSTA_stage_grouped_fit')\n    stage_labels = ['E12.5', 'E12.5']  # replace with all 26 manifest labels\n    Z_disk, W_stage, Ws, info = Run_Hierarchical_Multi_GPCA(\n        adatas=None, execution_mode='out_of_core', section_store=store_dir,\n        out_of_core_output_dir=output_dir, out_of_core_group_labels=stage_labels,\n        n_components=15, lambdas=0.5, rhos=0.20848642, max_iter=50, return_info=True)\n    print(info.converged, info.n_iter)\nelse:\n    print('Complete-atlas fit disabled; bundled assets come from the converged formal fit.')"""),
        md("""## 2. Spatially inspect aligned program coordinates\n\nThe scores below use the recorded stage alignment and common program rotation. Thus a given program label denotes one aligned loading coordinate across the displayed sections."""),
        code("""programs = ['P02', 'P04', 'P05']\nsections = scores[['section', 'stage']].drop_duplicates().itertuples(index=False)\nsections = list(sections)\nfig, axes = plt.subplots(len(programs), len(sections), figsize=(10.5, 9), constrained_layout=True)\nfor row, program in enumerate(programs):\n    lo, hi = scores[program].quantile([0.01, 0.99])\n    for col, item in enumerate(sections):\n        frame = scores.loc[scores.section.eq(item.section)]\n        points = axes[row, col].scatter(frame.x, frame.y, c=frame[program], s=1.2, cmap='RdBu_r', vmin=lo, vmax=hi, linewidths=0, rasterized=True)\n        axes[row, col].set(aspect='equal'); axes[row, col].invert_yaxis(); axes[row, col].axis('off')\n        if row == 0: axes[row, col].set_title(f'{item.stage}\\n{item.section}')\n        if col == 0: axes[row, col].text(-0.08, .5, program, transform=axes[row, col].transAxes, ha='right', va='center')\n    fig.colorbar(points, ax=axes[row, :], shrink=.6, label=f'{program} score')\nplt.show()"""),
        md("""## 3. Connect program labels to leading genes\n\nThe labels are post hoc summaries of aligned loading evidence. Annotation labels are not inputs to the factorization."""),
        code("""top_genes = pd.read_csv(ASSET_DIR / 'program_top_genes.csv')\nfig, axes = plt.subplots(1, 3, figsize=(11, 4), constrained_layout=True)\nfor ax, program in zip(axes, programs):\n    frame = top_genes.loc[top_genes.program.eq(program)].sort_values('loading')\n    ax.barh(frame.gene, frame.loading, color='#99000D')\n    ax.set_title(f'{program} positive loading genes'); ax.set_xlabel('loading')\nplt.show()"""),
        md("""## 4. Summarize aligned stage centres\n\nThis is a loading-space diagnostic: lower distances indicate more similar aligned stage centres. It is not a cell-level temporal trajectory or causal developmental analysis."""),
        code("""distance = pd.read_csv(ASSET_DIR / 'aligned_stage_loading_distances.csv').set_index('stage')\nplt.figure(figsize=(6.2, 5))\nsns.heatmap(distance, cmap='mako_r', square=True, cbar_kws={'label': 'Frobenius distance between aligned loading centres'})\nplt.title('Aligned stage-loading centre distances')\nplt.show()"""),
        md("""## 5. Section-level score dispersion\n\nThe displayed standard deviations summarize heterogeneity within each section. They are grouped by known stage for visualization only and do not fit or assert pseudotime."""),
        code("""activity = pd.read_csv(ASSET_DIR / 'section_program_activity.csv')\nplt.figure(figsize=(8, 4.5))\nsns.boxplot(data=activity, x='stage', y='sd', hue='program', palette=['#99000D', '#E34A33', '#F4A582'])\nplt.xlabel('Known sample stage'); plt.ylabel('Within-section score SD'); plt.title('Section-level program-score dispersion')\nplt.legend(title='Program', frameon=False)\nplt.show()"""),
        md("""## Checks and caveats\n\nThe bundled compact inputs are deterministic samples and tables from the converged formal stage-grouped fit (`n=2,322,560`, 26 sections, 2,000 genes). They are intended for visualization, not for recomputing the full atlas benchmark. This tutorial makes no P24 claim and does not report Monocle, Slingshot, pseudotime, or a causal developmental result."""),
    ]
    save_notebook(SOURCE / "Tutorial6_MOSTA_FAST26.ipynb", cells)


def set_abc_python_kernel():
    """Keep the inherited ABC tutorial executable in a standard Python kernel."""
    path = SOURCE / "Tutorial5_ABC_WMB.ipynb"
    notebook = nbf.read(path, as_version=4)
    notebook.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(notebook, path)


if __name__ == "__main__":
    if "--set-abc-kernel" in sys.argv:
        set_abc_python_kernel()
        print("Updated ABC tutorial kernel metadata.")
        raise SystemExit(0)
    build_merfish_assets()
    build_mosta_assets()
    write_merfish_notebook()
    write_mosta_notebook()
    print("Wrote compact real-result assets and executable notebooks.")
