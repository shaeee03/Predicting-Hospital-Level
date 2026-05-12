"""
================================================================================
SCRIPT 03: Preprocessing & Dimensionality Reduction
Project:   Predicting Number of Hospitals by Level (L1 / L2 / L3)
           per Municipality/City based on Socioeconomic &
           Infrastructural Factors
================================================================================

PIPELINE OVERVIEW
-----------------
  Step 1  Feature Selection
          Select the 30 model features (12 socioeconomic + 18 infrastructural)
          and 3 target variables from the merged Parquet file.

  Step 2  Missing-Value Imputation
          Poverty columns: ~5.6% missing → imputed with regional median
          Birth columns:   ~10.8% missing → imputed with regional median
          Regional median is preferred over global median to preserve
          geographic variation in these government-reported statistics.

  Step 3  Train / Test Split
          80/20 stratified split on a binned version of the composite
          hospital score (sum of L1+L2+L3) so that zero-hospital and
          high-hospital LGUs are represented in both sets.

  Step 4  Standardisation (Z-score)
          Fit on train, apply to train + test. Required before PCA because
          PCA is sensitive to feature scale. Population columns span
          hundreds to millions; infrastructure counts are 0–500+.
          Without standardisation, population would dominate all PCs.

  Step 5  PCA  (chosen over TruncatedSVD — see rationale below)
          Fit on standardised training features.
          Plot cumulative explained variance to select number of PCs.
          Plot loading biplots for the retained PCs.

WHY PCA AND NOT SVD/LSA?
--------------------------
  TruncatedSVD / LSA is recommended over PCA specifically for SPARSE
  matrices such as bag-of-words or TF-IDF representations, where
  mean-centering (required by PCA) would destroy sparsity and blow up
  memory. Our infrastructure columns are indeed zero-heavy (37–82%
  zeros) but they are NOT sparse in the technical sense — they are
  plain non-negative integer count columns in a 1,629 × 30 dense
  DataFrame. Mean-centering them does NOT create memory issues here.

  PCA is the correct choice because:
    (a) All features are numeric and dense after imputation.
    (b) Features differ wildly in scale → standardisation + PCA is
        the textbook approach (Shalizi §10, Aggarwal Ch. 2).
    (c) SVD on a non-mean-centred matrix conflates the mean with the
        first singular vector, producing a misleading "size" component.
    (d) We want interpretable principal components (PCA loadings map
        cleanly to original feature names), not latent semantic topics.
    (e) The professor's own PCA notebook uses the Cars dataset which
        has similar structure: mixed numeric features, dense matrix.

OUTPUTS (written to outputs/preprocessing/)
-------------------------------------------
  01_missing_summary.csv                   — missing-value audit
  02_imputation_check.csv                  — pre/post imputation stats
  03_train_test_split_summary.csv          — split sizes and target stats
  04_cumulative_explained_variance.png     — choose number of PCs
  05_biplot_PC1_PC2.png                    — loading biplot, PCs 1 & 2
  06_biplot_PC1_PC3.png                    — loading biplot, PCs 1 & 3
  07_biplot_PC2_PC3.png                    — loading biplot, PCs 2 & 3
  08_loadings_heatmap.png                  — full loading matrix heatmap
  X_train_scaled.parquet                   — standardised training features
  X_test_scaled.parquet                    — standardised test features
  X_train_pca.parquet                      — PCA-transformed training set
  X_test_pca.parquet                       — PCA-transformed test set
  y_train.parquet                          — training targets
  y_test.parquet                           — test targets
  pca_loadings.parquet                     — W matrix (PCs × features)
================================================================================
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR       = os.path.join(BASE_DIR, "outputs", "preprocessing")

DB_PATH       = os.path.join(PROCESSED_DIR, "hospital_data.db")
FINAL_XLSX    = os.path.join(BASE_DIR, "data", "clean", "final_dataset_clean.xlsx")

# ── Feature definitions ────────────────────────────────────────────────────
SOCIOECONOMIC_FEATURES = [
    "population_2020",
    "population_2024",
    "pop_growth_rate_pct",
    "poverty_incidence_2018_pct",
    "poverty_incidence_2021_pct",
    "poverty_incidence_2023_pct",
    "births_occurrence_both",
    "births_occurrence_male",
    "births_occurrence_female",
    "births_residence_both",
    "births_residence_male",
    "births_residence_female",
]

INFRASTRUCTURAL_FEATURES = [
    "atm",
    "bank",
    "bar",
    "bus_station",
    "cafe",
    "community_centre",
    "fast_food",
    "fuel",
    "parking",
    "pharmacy",
    "place_of_worship",
    "police",
    "post_office",
    "restaurant",
    "school",
    "shelter",
    "toilets",
    "townhall",
]

ALL_FEATURES = SOCIOECONOMIC_FEATURES + INFRASTRUCTURAL_FEATURES

TARGET_VARIABLES = [
    "hospital_count_level1",
    "hospital_count_level2",
    "hospital_count_level3",
]

POVERTY_COLS = [
    "poverty_incidence_2018_pct",
    "poverty_incidence_2021_pct",
    "poverty_incidence_2023_pct",
]

BIRTH_COLS = [
    "births_occurrence_both",
    "births_occurrence_male",
    "births_occurrence_female",
    "births_residence_both",
    "births_residence_male",
    "births_residence_female",
]

# ── Plotting style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
})

PALETTE = sns.color_palette("tab10")


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS  (adapted from prof's utils.py)
# ═══════════════════════════════════════════════════════════════════════════

def plot_cum_exp_var(exp_var_ratio: np.ndarray, tol: float = 0.90):
    """
    Plot cumulative explained variance vs number of PCs.

    Adapted directly from prof's utils.plot_cum_exp_var.
    Returns fig, ax, and the PC threshold at the given tolerance.
    """
    exp_var = exp_var_ratio.cumsum()
    thresh = int(np.min(np.arange(len(exp_var))[exp_var >= tol]) + 1)

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)

    ax.plot(range(0, len(exp_var) + 1), [0] + exp_var.tolist(),
            lw=3.0, marker="o", color=PALETTE[0])
    ax.axvline(thresh, linestyle="-", lw=2.5, color="tab:orange")
    ax.axhline(tol, linestyle="--", lw=1.5, color="grey", alpha=0.6)

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(thresh + 0.15, 0.07,
            f"Number of PCs: {thresh}  ({tol*100:.0f}% threshold)",
            color="tab:orange", weight="bold", fontsize=12, transform=trans)

    ax.set_xlim(0, len(exp_var))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Number of Components", fontsize=13)
    ax.set_ylabel("Cumulative Explained Variance", fontsize=13)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("Cumulative Explained Variance versus Number of PCs",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    return fig, ax, thresh


def plot_principal_components(X_transformed: pd.DataFrame,
                               W: pd.DataFrame,
                               column_1: str,
                               column_2: str,
                               margin: float = 0.05,
                               hue: np.ndarray = None,
                               hue_label: str = "Group",
                               vals=None,
                               figsize=(17, 8),
                               wspace: float = 0.06,
                               palette="default"):
    """
    Side-by-side scatter plot of LGU scores and loading biplot.

    Left panel  — LGU coordinates projected onto (column_1, column_2)
    Right panel — top-20 feature loading vectors in PC-space

    Adapted from prof's utils.plot_principal_components.
    """
    if palette == "default":
        palette = sns.color_palette("tab10")

    fig, axes = plt.subplots(1, 2, figsize=figsize,
                             gridspec_kw={"wspace": wspace})

    # ── Left: LGU score scatter ───────────────────────────────────────────
    if hue is not None:
        if vals is None:
            vals = sorted(set(hue))
        for i, val in enumerate(vals):
            mask = hue == val
            axes[0].plot(X_transformed.loc[mask, column_1],
                         X_transformed.loc[mask, column_2],
                         "o", color=palette[i % len(palette)],
                         alpha=0.55, ms=4, label=val)
        axes[0].legend(title=hue_label, fontsize=9, title_fontsize=9,
                        markerscale=1.5)
    else:
        axes[0].plot(X_transformed.loc[:, column_1],
                     X_transformed.loc[:, column_2],
                     "o", color=palette[0], alpha=0.45, ms=4)

    axes[0].set_xlabel(column_1, fontsize=12)
    axes[0].set_ylabel(column_2, fontsize=12)
    axes[0].set_title("LGU Scores", fontsize=12, weight="bold")
    for spine in ["top", "right"]:
        axes[0].spines[spine].set_visible(False)

    # ── Right: loading biplot ─────────────────────────────────────────────
    W_T = W.T
    lsas = np.column_stack([
        W_T.loc[:, column_1].values,
        W_T.loc[:, column_2].values,
    ])
    weights = np.linalg.norm(lsas, axis=1)
    top_idx = weights.argsort()[-20:]
    features = W_T.index.tolist()

    for feat, vec in zip(np.array(features)[top_idx], lsas[top_idx]):
        axes[1].annotate(
            "", xy=(vec[0], vec[1]), xycoords="data",
            xytext=(0, 0), textcoords="data",
            arrowprops=dict(facecolor=palette[0], edgecolor="none",
                            width=1.5, headwidth=7))
        axes[1].text(vec[0], vec[1], feat, ha="center",
                     color=palette[1], fontsize=10, weight="bold", zorder=10)

    xlim = [W_T[column_1].min(), W_T[column_1].max()]
    ylim = [W_T[column_2].min(), W_T[column_2].max()]
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    axes[1].set_xlim(xlim[0] - xr * margin, xlim[1] + xr * margin)
    axes[1].set_ylim(ylim[0] - yr * margin, ylim[1] + yr * margin)
    axes[1].tick_params(axis="both", which="both",
                         top=False, bottom=False, left=False,
                         labelbottom=False, labelleft=False)
    axes[1].set_xlabel(column_1, fontsize=12)
    axes[1].set_ylabel(column_2, fontsize=12)
    axes[1].set_title("Feature Loadings (top-20 by magnitude)", fontsize=12,
                       weight="bold")
    for spine in ["top", "right", "left", "bottom"]:
        axes[1].spines[spine].set_visible(False)

    return fig, axes


def plot_loadings_heatmap(W: pd.DataFrame, n_pcs: int) -> plt.Figure:
    """
    Heatmap of the full PC-loading matrix for retained PCs.

    Positive loadings are blue (feature contributes positively to the PC).
    Negative loadings are red (feature contributes negatively).
    """
    W_retained = W.iloc[:n_pcs]
    fig, ax = plt.subplots(figsize=(14, 0.55 * n_pcs + 2.5))
    sns.heatmap(
        W_retained,
        annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        linewidths=0.4, linecolor="white",
        annot_kws={"size": 8},
        ax=ax,
    )
    ax.set_title(f"PCA Loading Matrix — Retained {n_pcs} PCs",
                 fontsize=13, weight="bold", pad=10)
    ax.set_xlabel("Original Feature", fontsize=11)
    ax.set_ylabel("Principal Component", fontsize=11)
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# STEP 1 — Feature Selection
# ═══════════════════════════════════════════════════════════════════════════

def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract X (features) and y (targets) from the merged DataFrame.

    Returns
    -------
    X : DataFrame, shape (n_lgu, 30)   — raw, unimputed features
    y : DataFrame, shape (n_lgu, 3)    — target hospital counts
    """
    print("\n[1/5] Feature Selection")
    missing_feats = [c for c in ALL_FEATURES if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing feature columns in dataset: {missing_feats}")

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_VARIABLES].copy()

    print(f"  Features : {X.shape[1]} ({len(SOCIOECONOMIC_FEATURES)} socioeconomic "
          f"+ {len(INFRASTRUCTURAL_FEATURES)} infrastructural)")
    print(f"  Samples  : {X.shape[0]} LGUs")
    print(f"  Targets  : {y.shape[1]} (L1 / L2 / L3 hospital counts)")

    # Save missing-value audit
    miss = X.isnull().sum().to_frame("missing_count")
    miss["missing_pct"] = (miss["missing_count"] / len(X) * 100).round(2)
    miss["feature_group"] = [
        "socioeconomic" if c in SOCIOECONOMIC_FEATURES else "infrastructural"
        for c in miss.index
    ]
    miss.to_csv(os.path.join(OUT_DIR, "01_missing_summary.csv"))
    print(f"\n  Missing-value audit saved → 01_missing_summary.csv")
    print(f"  Poverty columns  missing: "
          f"{X[POVERTY_COLS].isnull().any(axis=1).sum()} LGUs "
          f"({X[POVERTY_COLS].isnull().any(axis=1).mean()*100:.1f}%)")
    print(f"  Birth columns    missing: "
          f"{X[BIRTH_COLS].isnull().any(axis=1).sum()} LGUs "
          f"({X[BIRTH_COLS].isnull().any(axis=1).mean()*100:.1f}%)")
    print(f"  Infrastructure   missing: 0 LGUs (complete)")

    return X, y


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2 — Missing-Value Imputation
# ═══════════════════════════════════════════════════════════════════════════

def impute(X: pd.DataFrame, region: pd.Series) -> pd.DataFrame:
    """
    Impute missing values using regional medians.

    RATIONALE
    ---------
    Poverty incidence and birth counts are government-reported statistics
    that vary substantially by region (e.g. BARMM >> NCR). Global median
    imputation would systematically underestimate poverty in poor regions
    and overestimate it in wealthy ones. Regional median preserves this
    geographic heterogeneity, which is exactly what the model needs to
    learn from.

    Columns imputed
    ---------------
    poverty_incidence_{2018,2021,2023}_pct — ~5.6% missing
    births_{occurrence,residence}_{both,male,female} — ~10.8% missing

    Infrastructure columns (atm, bank, …) — complete; no imputation needed.
    """
    print("\n[2/5] Imputation (regional median)")

    impute_cols = POVERTY_COLS + BIRTH_COLS
    X = X.copy()

    # Record pre-imputation stats for audit
    pre_stats = X[impute_cols].describe().T[["mean", "50%", "std"]]
    pre_stats.columns = ["pre_mean", "pre_median", "pre_std"]

    # Impute: for each column, fill NaN with median of same region
    for col in impute_cols:
        before = X[col].isnull().sum()
        regional_median = X.groupby(region)[col].transform("median")
        # Fallback to global median for any LGU whose entire region is NaN
        global_median = X[col].median()
        X[col] = X[col].fillna(regional_median).fillna(global_median)
        after = X[col].isnull().sum()
        print(f"  {col:<38}  NaN: {before:>4} → {after}")

    post_stats = X[impute_cols].describe().T[["mean", "50%", "std"]]
    post_stats.columns = ["post_mean", "post_median", "post_std"]
    audit = pd.concat([pre_stats, post_stats], axis=1)
    audit.to_csv(os.path.join(OUT_DIR, "02_imputation_check.csv"))
    print(f"\n  Imputation audit saved → 02_imputation_check.csv")

    return X


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3 — Train / Test Split
# ═══════════════════════════════════════════════════════════════════════════

def split(X: pd.DataFrame,
          y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                     pd.DataFrame, pd.DataFrame]:
    """
    80/20 stratified train-test split.

    STRATIFICATION STRATEGY
    -----------------------
    The targets are heavily zero-skewed (>73% of LGUs have zero hospitals
    at every level). A random split could leave all high-value LGUs in one
    set. We stratify on a 4-bin composite score:
        score = L1_count + L2_count + L3_count
        bins  = [0, 1, 2, 4, max+1)  →  "zero", "low", "medium", "high"
    This ensures the rare high-hospital LGUs appear in both train and test.
    """
    print("\n[3/5] Train / Test Split (80/20 stratified)")

    composite = y.sum(axis=1)
    bins = [0, 1, 2, 4, composite.max() + 1]
    labels = ["zero", "low", "medium", "high"]
    strat_bin = pd.cut(composite, bins=bins, labels=labels,
                       include_lowest=True, right=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=strat_bin
    )

    summary = pd.DataFrame({
        "split": ["train", "test"],
        "n_lgu": [len(X_train), len(X_test)],
        "l1_mean": [y_train["hospital_count_level1"].mean(),
                    y_test["hospital_count_level1"].mean()],
        "l2_mean": [y_train["hospital_count_level2"].mean(),
                    y_test["hospital_count_level2"].mean()],
        "l3_mean": [y_train["hospital_count_level3"].mean(),
                    y_test["hospital_count_level3"].mean()],
        "pct_zero_l1": [(y_train["hospital_count_level1"] == 0).mean(),
                        (y_test["hospital_count_level1"] == 0).mean()],
    })
    summary.to_csv(os.path.join(OUT_DIR, "03_train_test_split_summary.csv"),
                   index=False)
    print(f"  Train : {len(X_train)} LGUs")
    print(f"  Test  : {len(X_test)} LGUs")
    print(f"  Split summary saved → 03_train_test_split_summary.csv")

    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4 — Standardisation
# ═══════════════════════════════════════════════════════════════════════════

def standardise(X_train: pd.DataFrame,
                X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
                                                StandardScaler]:
    """
    Z-score standardisation: subtract mean, divide by std.

    Fit on training set only; apply same transform to test set.
    This prevents test-set information from leaking into the scaler.

    WHY STANDARDISE BEFORE PCA?
    ----------------------------
    PCA maximises variance. Population_2020 has std ≈ 200,000 while
    bar (# of bars) has std ≈ 8. Without standardisation, PCA would
    almost entirely describe population variation and the 28 other
    features would contribute virtually nothing. Z-scoring puts all
    features on a common unit-variance scale so that PCA reflects
    genuine multi-feature structure.
    """
    print("\n[4/5] Standardisation (Z-score, fit on train only)")

    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    print(f"  Train — mean ≈ {X_train_sc.mean().mean():.2e} "
          f"(should be ~0), std ≈ {X_train_sc.std().mean():.3f} (should be ~1)")
    print(f"  Test  — mean ≈ {X_test_sc.mean().mean():.2e}, "
          f"std ≈ {X_test_sc.std().mean():.3f}")

    return X_train_sc, X_test_sc, scaler


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5 — PCA
# ═══════════════════════════════════════════════════════════════════════════

def run_pca(X_train_sc: pd.DataFrame,
            X_test_sc: pd.DataFrame,
            y_train: pd.DataFrame,
            tol: float = 0.90) -> dict:
    """
    Fit PCA on standardised training features, produce all required plots,
    and return transformed DataFrames.

    Plots produced
    --------------
    04_cumulative_explained_variance.png
        Used to select n_components — the first PC that pushes cumulative
        explained variance past the 90% threshold.

    05_biplot_PC1_PC2.png
        LGU score scatter + top-20 feature loading vectors for PCs 1 & 2.
        Coloured by region to reveal geographic clustering.

    06_biplot_PC1_PC3.png, 07_biplot_PC2_PC3.png
        Same for other pairs of retained PCs, if n_components >= 3.

    08_loadings_heatmap.png
        Heatmap of the full loading matrix W for the retained PCs.
        Enables systematic interpretation of which original features
        align most strongly with each PC.

    Returns
    -------
    dict with keys: pca, n_components, W, X_train_pca, X_test_pca
    """
    print("\n[5/5] PCA")

    # ── Fit full PCA (all components) to get explained variance curve ──────
    pca_full = PCA()
    pca_full.fit(X_train_sc)

    exp_var_ratio = pca_full.explained_variance_ratio_
    print(f"\n  Explained variance per PC (first 10):")
    for i, ev in enumerate(exp_var_ratio[:10], 1):
        bar = "█" * int(ev * 50)
        print(f"    PC{i:>2}: {ev*100:5.2f}%  {bar}")

    # ── Plot 1: Cumulative explained variance — choose n_components ────────
    print(f"\n  Plotting cumulative explained variance (tol={tol:.0%}) ...")
    fig_cev, ax_cev, n_components = plot_cum_exp_var(exp_var_ratio, tol=tol)
    _save(fig_cev, "04_cumulative_explained_variance.png")

    print(f"\n  Selected n_components = {n_components}")
    print(f"  Cumulative variance explained: "
          f"{exp_var_ratio[:n_components].sum()*100:.2f}%")
    print()
    print("  DECISION RATIONALE:")
    print(f"  The cumulative explained variance curve crosses the 90%")
    print(f"  threshold at PC {n_components}. Retaining {n_components} PCs")
    print(f"  captures {exp_var_ratio[:n_components].sum()*100:.1f}% of the")
    print(f"  total variance in the 30 standardised features, reducing")
    print(f"  dimensionality from 30 to {n_components} while preserving")
    print(f"  the majority of the information. Components beyond PC {n_components}")
    print(f"  each explain <{exp_var_ratio[n_components]*100:.1f}% and represent")
    print(f"  noise or idiosyncratic variation unlikely to generalise.")

    # ── Re-fit PCA with selected n_components ─────────────────────────────
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca_arr = pca.fit_transform(X_train_sc)
    X_test_pca_arr  = pca.transform(X_test_sc)

    pc_labels = [f"PC {i+1}" for i in range(n_components)]
    X_train_pca = pd.DataFrame(X_train_pca_arr, columns=pc_labels,
                                index=X_train_sc.index)
    X_test_pca  = pd.DataFrame(X_test_pca_arr,  columns=pc_labels,
                                index=X_test_sc.index)

    # Loading matrix W: shape (n_components, n_features)
    W = pd.DataFrame(pca.components_,
                     columns=X_train_sc.columns,
                     index=pc_labels)

    # ── Plot 2–4: Biplots for PC pairs ─────────────────────────────────────
    # Colour LGUs by L1 hospital presence (0 vs ≥1) for interpretability
    hue_arr = (y_train["hospital_count_level1"] >= 1).map(
        {False: "No L1 hospital", True: "Has L1 hospital"}
    ).values

    biplot_pairs = [("PC 1", "PC 2", "05_biplot_PC1_PC2.png")]
    if n_components >= 3:
        biplot_pairs += [
            ("PC 1", "PC 3", "06_biplot_PC1_PC3.png"),
            ("PC 2", "PC 3", "07_biplot_PC2_PC3.png"),
        ]

    for col1, col2, fname in biplot_pairs:
        print(f"  Plotting biplot {col1} vs {col2} ...")
        fig_bp, axes_bp = plot_principal_components(
            X_train_pca, W, col1, col2,
            hue=hue_arr,
            hue_label="L1 Hospital",
            palette=[PALETTE[0], PALETTE[1]],
        )
        fig_bp.suptitle(f"PCA Biplot — {col1} vs {col2}",
                         fontsize=14, weight="bold", y=1.01)
        _save(fig_bp, fname)

    # ── Plot 5: Loadings heatmap ───────────────────────────────────────────
    print(f"  Plotting loading heatmap for {n_components} PCs ...")
    fig_heat = plot_loadings_heatmap(W, n_components)
    _save(fig_heat, "08_loadings_heatmap.png")

    # ── Print loading interpretation summary ───────────────────────────────
    print(f"\n  Top-3 positive and negative loadings per retained PC:")
    for pc in pc_labels:
        row = W.loc[pc].sort_values()
        top_neg = row.head(3)
        top_pos = row.tail(3)[::-1]
        print(f"\n    {pc}  ({pca.explained_variance_ratio_[int(pc.split()[1])-1]*100:.1f}%):")
        print(f"      Positive (↑{pc} means more of):  "
              + ", ".join(f"{f} ({v:+.2f})" for f, v in top_pos.items()))
        print(f"      Negative (↑{pc} means less of): "
              + ", ".join(f"{f} ({v:+.2f})" for f, v in top_neg.items()))

    return {
        "pca": pca,
        "n_components": n_components,
        "W": W,
        "X_train_pca": X_train_pca,
        "X_test_pca":  X_test_pca,
    }


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to OUT_DIR and close it."""
    path = os.path.join(OUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"    Saved → {filename}")


def _save_parquet(df: pd.DataFrame, filename: str) -> None:
    """
    Save a DataFrame to data/processed/ as Parquet (snappy-compressed).
    Falls back to CSV if neither pyarrow nor fastparquet is installed.
    Install with: pip install pyarrow
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, filename)
    try:
        df.to_parquet(path, index=True, compression="snappy")
        print(f"  Saved → {filename}  ({df.shape[0]} × {df.shape[1]})")
    except ImportError:
        csv_name = filename.replace(".parquet", ".csv")
        csv_path = os.path.join(PROCESSED_DIR, csv_name)
        df.to_csv(csv_path, index=True)
        print(f"  ⚠  pyarrow not installed — saved as CSV: {csv_name}")
        print(f"     Install with:  pip install pyarrow")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def _load_dataset() -> pd.DataFrame:
    """
    Load the merged LGU dataset from the SQLite database produced by
    Script 02 (02_storage.py), with a fallback to the source XLSX if
    the database has not been built yet.

    Search order
    ------------
    1. data/processed/hospital_data.db  (lgu_merged table)  [preferred]
    2. data/clean/final_dataset_clean.xlsx                   [fallback]
    """
    import sqlite3

    # ── Try SQLite first ──────────────────────────────────────────────────
    if os.path.exists(DB_PATH):
        print(f"  Reading lgu_merged from {os.path.basename(DB_PATH)} ...")
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM lgu_merged", conn)
        conn.close()

        missing = [c for c in INFRASTRUCTURAL_FEATURES if c not in df.columns]
        if not missing:
            print(f"  OK  Loaded {len(df)} LGUs x {len(df.columns)} cols from SQLite")
            return df
        print(f"    Skipping DB — missing infra columns (e.g. {missing[:3]})")

    # ── Fallback: XLSX ────────────────────────────────────────────────────
    if os.path.exists(FINAL_XLSX):
        print(f"  Reading {os.path.basename(FINAL_XLSX)} (run 02_storage.py "
              f"first to use SQLite) ...")
        df = pd.read_excel(FINAL_XLSX)
        missing = [c for c in INFRASTRUCTURAL_FEATURES if c not in df.columns]
        if not missing:
            print(f"  OK  Loaded {len(df)} LGUs x {len(df.columns)} cols from XLSX")
            return df

    raise FileNotFoundError(
        "Could not find a usable dataset. Ensure one of the following exists:\n"
        f"  1. {DB_PATH} (run 02_storage.py)\n"
        f"  2. {FINAL_XLSX}"
    )


def main() -> None:
    print("=" * 70)
    print("PREDICTING NUMBER OF HOSPITALS — SCRIPT 03: PREPROCESSING")
    print("Dimensionality Reduction: PCA (over TruncatedSVD/LSA)")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load — tries SQLite (lgu_merged table) first, falls back to XLSX
    df = _load_dataset()

    # Normalise the province column name (Script 01 outputs 'province_x'
    # when the join produces a suffix; rename to 'province' for consistency)
    if "province_x" in df.columns and "province" not in df.columns:
        df = df.rename(columns={"province_x": "province"})

    region_col = df["region"] if "region" in df.columns else None

    # Pipeline
    X, y             = select_features(df)
    X_imp            = impute(X, region_col)
    X_train, X_test, y_train, y_test = split(X_imp, y)
    X_train_sc, X_test_sc, _ = standardise(X_train, X_test)
    results          = run_pca(X_train_sc, X_test_sc, y_train, tol=0.90)

    # Persist all outputs
    print("\n  Persisting train/test splits and PCA outputs ...")
    _save_parquet(X_train_sc,              "X_train_scaled.parquet")
    _save_parquet(X_test_sc,               "X_test_scaled.parquet")
    _save_parquet(results["X_train_pca"],  "X_train_pca.parquet")
    _save_parquet(results["X_test_pca"],   "X_test_pca.parquet")
    _save_parquet(y_train,                 "y_train.parquet")
    _save_parquet(y_test,                  "y_test.parquet")
    _save_parquet(results["W"],            "pca_loadings.parquet")

    print("\n" + "=" * 70)
    print("DONE.")
    print(f"  Plots     → outputs/preprocessing/")
    print(f"  Data      → data/processed/  (Parquet)")
    print(f"  PCs retained  : {results['n_components']}")
    print(f"  Variance explained: "
          f"{results['pca'].explained_variance_ratio_.sum()*100:.2f}%")
    print("\nNext step: run 04_model.py")
    print("=" * 70)


if __name__ == "__main__":
    main()