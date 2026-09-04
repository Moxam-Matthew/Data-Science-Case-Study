"""
06_interpretation.py -- Making the models sayable.

A model a clinician cannot interrogate does not get used, whatever its AUC.
Three techniques, in descending order of how much they can be trusted.

    Run:  python 06_interpretation.py

THE QUESTIONS
    Q28  SHAP explains an XGBoost prediction as a sum of per-feature
         contributions. A clinician will read those as odds ratios and as
         causes. Both readings are wrong. What is a SHAP value actually?
    Q29  Hierarchical clustering finds patient subtypes without using the
         outcome. Single linkage, complete linkage and Ward's method applied to
         the same patients give different answers. Which is right, and what
         would make any of them a phenotype?
    Q30  The depth-3 tree is the only model in this project genuinely beaten
         on discrimination -- and it is better calibrated than XGBoost. Make
         the case for shipping it anyway, and then the case against.

Author: Matthew Moxam
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import viz
from modelling import (
    CV_REPEATS,
    HORIZON_DAYS,
    calibration_metrics,
    cross_val_predictions,
    OUTCOME_LABEL,
    RANDOM_STATE,
    build_pipeline,
    default_predictors,
    make_outcome,
)
from report import Facts, RULE, configure_pandas, header, question, render_answers, run_and_capture
from support2 import analysis_frames

OUT_DIR = Path(__file__).resolve().parent / "output"
LINKAGES = ["single", "complete", "average", "ward"]
N_CLUSTERS = 4


# ═══ Shared fitted objects ═══════════════════════════════════════════════════
def fit_xgb(chf: pd.DataFrame, predictors: list[str], y: np.ndarray):
    from xgboost import XGBClassifier

    pipe = build_pipeline(
        chf, predictors,
        XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                      min_child_weight=5, eval_metric="logloss",
                      random_state=RANDOM_STATE, n_jobs=-1),
        scale=False)
    pipe.fit(chf[predictors], y)
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    Z = pd.DataFrame(
        pipe.named_steps["prep"].transform(
            pipe.named_steps["indicators"].transform(chf[predictors])),
        columns=names, index=chf.index)
    return pipe, Z, names


def design_matrix(chf: pd.DataFrame, predictors: list[str], y: np.ndarray):
    """Imputed and scaled feature matrix, for clustering."""
    from sklearn.linear_model import LogisticRegression

    pipe = build_pipeline(chf, predictors,
                          LogisticRegression(max_iter=1000))
    pipe.fit(chf[predictors], y)
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    Z = pd.DataFrame(
        pipe.named_steps["prep"].transform(
            pipe.named_steps["indicators"].transform(chf[predictors])),
        columns=names, index=chf.index)
    return Z


# ═══ Q28. SHAP ═══════════════════════════════════════════════════════════════
def compute_shap(pipe, Z: pd.DataFrame, y: np.ndarray) -> dict:
    import shap

    model = pipe.named_steps["model"]
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(Z)
    mean_abs = pd.Series(np.abs(values).mean(axis=0), index=Z.columns)
    ranked = mean_abs.sort_values(ascending=False)

    # One patient, to show what a local explanation looks like.
    idx = int(np.argmax(np.abs(values).sum(axis=1)))
    local = pd.Series(values[idx], index=Z.columns).sort_values(
        key=np.abs, ascending=False).head(8)
    return {"values": values, "ranked": ranked, "local": local,
            "local_idx": idx, "base": float(explainer.expected_value),
            "patient_id": Z.index[idx]}


def report_shap(r: dict, Z: pd.DataFrame) -> None:
    question(28, "SHAP explains an XGBoost prediction as a sum of per-feature\n"
                 "contributions. A clinician will read those as odds ratios and as\n"
                 "causes. Both readings are wrong. What is a SHAP value actually?")
    print("  Global importance -- mean |SHAP| across the training cohort")
    print("  (log-odds units, NOT odds ratios):\n")
    top = r["ranked"].head(12)
    for name, v in top.items():
        bar = "#" * int(v / top.iloc[0] * 34)
        print(f"    {name:<20} {bar:<34} {v:.4f}")

    print(f"\n  Local explanation for one patient (id {r['patient_id']}),")
    print(f"  the largest total attribution in the cohort.")
    print(f"    baseline log-odds {r['base']:+.3f}")
    for name, v in r["local"].items():
        raw = Z.loc[r["patient_id"], name]
        print(f"    {name:<20} value {raw:>8.2f}   contribution {v:+.3f}")
    print(f"    {'sum':<20} {'':>14}   {r['local'].sum():+.3f}")


# ═══ Q29. Hierarchical clustering ════════════════════════════════════════════
def compute_clustering(Z: pd.DataFrame, y: np.ndarray) -> dict:
    from scipy.cluster.hierarchy import cophenet, fcluster, linkage
    from scipy.spatial.distance import pdist
    from sklearn.metrics import silhouette_score

    D = pdist(Z.values, metric="euclidean")
    results = {}
    for method in LINKAGES:
        L = linkage(D, method=method)
        coph, _ = cophenet(L, D)
        labels = fcluster(L, N_CLUSTERS, criterion="maxclust")
        sizes = pd.Series(labels).value_counts().sort_index()
        sil = (silhouette_score(Z.values, labels)
               if len(np.unique(labels)) > 1 else np.nan)
        # Does the partition separate the outcome at all?
        mort = pd.Series(y).groupby(labels).mean() * 100
        results[method] = {"linkage": L, "cophenetic": float(coph),
                           "labels": labels, "sizes": sizes,
                           "silhouette": float(sil),
                           "largest_share": float(sizes.max() / len(labels) * 100),
                           "mortality": mort}
    return results


def report_clustering(r: dict, y: np.ndarray) -> None:
    question(29, "Hierarchical clustering finds patient subtypes without using the\n"
                 "outcome. Single, complete, average and Ward linkage applied to the\n"
                 "same patients give different answers. Which is right, and what\n"
                 "would make any of them a phenotype?")
    rows = []
    for method, d in r.items():
        rows.append({
            "linkage": method,
            "cophenetic_corr": d["cophenetic"],
            "silhouette": d["silhouette"],
            "largest_cluster_pct": d["largest_share"],
            "cluster_sizes": ", ".join(str(int(v)) for v in d["sizes"]),
            "mortality_range_pp": d["mortality"].max() - d["mortality"].min(),
        })
    t = pd.DataFrame(rows)
    for c in ("cophenetic_corr", "silhouette", "largest_cluster_pct",
              "mortality_range_pp"):
        t[c] = t[c].round(3)
    print(t.to_string(index=False))
    print(f"\n  Cohort mortality for reference: {y.mean()*100:.1f}%")
    print("\n  cophenetic_corr: how faithfully the tree preserves the original")
    print("    pairwise distances. Higher is a better summary of the data.")
    print("  silhouette: cluster separation, -1 to 1. Near 0 means the clusters")
    print("    are not separated in feature space.")
    print("  largest_cluster_pct: a value near 100 is the chaining failure --")
    print("    one giant cluster plus a handful of singletons.")


# ═══ Q30. The tree as a rule ═════════════════════════════════════════════════
def compute_tree(chf: pd.DataFrame, predictors: list[str], y: np.ndarray) -> dict:
    from sklearn.tree import DecisionTreeClassifier, export_text

    pipe = build_pipeline(chf, predictors,
                          DecisionTreeClassifier(max_depth=3, min_samples_leaf=40,
                                                 random_state=RANDOM_STATE),
                          scale=False)
    pipe.fit(chf[predictors], y)
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    tree = pipe.named_steps["model"]
    Z = pd.DataFrame(
        pipe.named_steps["prep"].transform(
            pipe.named_steps["indicators"].transform(chf[predictors])),
        columns=names, index=chf.index)
    leaves = tree.apply(Z.values)
    leaf_stats = (pd.DataFrame({"leaf": leaves, "y": y})
                  .groupby("leaf").agg(n=("y", "size"), events=("y", "sum"),
                                       risk=("y", "mean")))
    leaf_stats["risk"] = leaf_stats.risk * 100
    return {"pipe": pipe, "tree": tree, "names": names, "Z": Z,
            "text": export_text(tree, feature_names=names, decimals=1),
            "leaves": leaf_stats.sort_values("risk"),
            "n_leaves": int(tree.get_n_leaves())}


def report_tree(r: dict, y: np.ndarray) -> None:
    question(30, "The depth-3 tree is the only model here genuinely beaten on\n"
                 "discrimination, yet it is better calibrated than XGBoost. Make\n"
                 "the case for shipping it anyway, and then the case against.")
    print(f"  {r['n_leaves']} terminal groups. The whole model:\n")
    for line in r["text"].splitlines():
        print("    " + line)
    print("\n  Risk by terminal group:")
    print(r["leaves"].round(1).to_string())
    print(f"\n  Cohort risk {y.mean()*100:.1f}%. Spread across groups: "
          f"{r['leaves'].risk.min():.1f}% to {r['leaves'].risk.max():.1f}%.")


# ═══ Figures ═════════════════════════════════════════════════════════════════
def figure_shap(shap_r: dict, Z: pd.DataFrame):
    import matplotlib.pyplot as plt

    top = shap_r["ranked"].head(14).iloc[::-1]
    vals = shap_r["values"]
    cols = [Z.columns.get_loc(c) for c in top.index]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    ax1.barh(top.index, top.values, color=viz.SERIES_BLUE, height=0.7)
    ax1.set_xlabel("Mean |SHAP| (log-odds)")
    ax1.set_title("Global: which features move predictions")
    ax1.grid(axis="y", visible=False)
    viz.despine(ax1)

    rng = np.random.default_rng(RANDOM_STATE)
    for i, c in enumerate(cols):
        v = vals[:, c]
        jitter = rng.normal(0, 0.11, len(v))
        feat = Z.iloc[:, c].values
        rank = pd.Series(feat).rank(pct=True).values
        ax2.scatter(v, i + jitter, c=rank, cmap=viz.sequential_cmap(),
                    s=7, alpha=0.55, linewidths=0)
    ax2.axvline(0, color=viz.BASELINE, lw=1.2)
    ax2.set_yticks(range(len(cols)), top.index, fontsize=9)
    ax2.set_xlabel("SHAP value (log-odds contribution)")
    ax2.set_title("Local: direction and spread, coloured by feature value")
    ax2.grid(axis="y", visible=False)
    viz.despine(ax2)
    sm = plt.cm.ScalarMappable(cmap=viz.sequential_cmap())
    sm.set_array([])   # colorbar needs an array even when the mapping is manual
    cb = fig.colorbar(sm, ax=ax2, shrink=0.6, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("feature value (percentile)", fontsize=8.5)
    cb.outline.set_visible(False)

    fig.tight_layout()
    viz.caption(fig, f"XGBoost fitted on the CHF training cohort, {OUTCOME_LABEL}. Units are log-odds\n"
                     f"contributions, not odds ratios, and the attribution is to the MODEL's use of a\n"
                     f"feature -- not to a causal effect on the patient.", y=-0.06)
    return viz.save(fig, "13_shap.png")


def figure_dendrogram(clust: dict, Z: pd.DataFrame):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram, set_link_color_palette

    set_link_color_palette([viz.SERIES_BLUE, viz.SERIES_ORANGE,
                            viz.SERIES[2], viz.SERIES[3]])
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, method in zip(axes.ravel(), LINKAGES):
        d = clust[method]
        dendrogram(d["linkage"], ax=ax, truncate_mode="lastp", p=25,
                   no_labels=True, color_threshold=d["linkage"][-(N_CLUSTERS-1), 2],
                   above_threshold_color=viz.INK_MUTED)
        ax.set_title(f"{method} linkage    cophenetic r={d['cophenetic']:.2f}   "
                     f"silhouette={d['silhouette']:.2f}", fontsize=10)
        ax.set_ylabel("Distance", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(False)
        viz.despine(ax, keep=("left",))
    set_link_color_palette(None)
    fig.suptitle("Same patients, four linkage criteria, four different answers",
                 fontsize=12, fontweight="600")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    viz.caption(fig, f"CHF training cohort, n={len(Z):,}, imputed and standardised predictors, Euclidean\n"
                     f"distance, last 25 merges shown. Single, complete and average linkage all chain into one\n"
                     f"cluster holding 98-99% of patients; Ward is the only criterion that divides the cohort at\n"
                     f"all, and even it leaves 58% in one group. None of this makes the partition a phenotype.")
    return viz.save(fig, "14_dendrogram.png")


def figure_tree(r: dict):
    import matplotlib.pyplot as plt
    from sklearn.tree import plot_tree

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_tree(r["tree"], feature_names=r["names"],
              class_names=[f"alive at {HORIZON_DAYS}d", f"dead by {HORIZON_DAYS}d"],
              filled=True, rounded=True, impurity=False, proportion=True,
              fontsize=9, ax=ax,
              node_ids=False, precision=2)
    ax.set_title("The bedside rule: three questions, six groups", fontsize=12,
                 fontweight="600")
    viz.caption(fig, f"CHF training cohort. Loses to every other model on discrimination and is the only\n"
                     f"one that fits on a card. Values are proportions; shading is predicted class.", y=0.02)
    return viz.save(fig, "15_decision_tree.png")


ANSWERS = """
ANSWERS
{rule}

A28. WHAT A SHAP VALUE IS, AND THE TWO MISREADINGS
    A SHAP value is the average marginal contribution of a feature to THIS
    prediction, over all orderings in which features could be added to the
    model. It is an attribution of the model's output, and its units here are
    log-odds. The values for one patient sum exactly to the difference between
    that patient's prediction and the cohort baseline, which is why the local
    explanation above adds up.

    The first misreading is as an odds ratio. It is not one. An odds ratio is a
    single number describing a feature's effect everywhere; a SHAP value is
    specific to one patient and changes from patient to patient, which is
    precisely what makes it useful for a bedside explanation and useless for a
    Table 2. If a clinician asks "so what is the odds ratio for creatinine", the
    honest answer is that this model does not have one, and that the elastic net
    in 05_modelling.py does.

    The second misreading is causal, and it is the dangerous one. SHAP explains
    the MODEL, not the patient. If two features are correlated the attribution
    between them is close to arbitrary -- the model may lean on either, and SHAP
    will faithfully report whichever it used. Nothing here licenses the sentence
    "lowering this value would reduce risk". A high contribution means the model
    used the feature, not that the feature causes the outcome.

    Stated positively, and this is the real reason to include it: SHAP gives a
    per-patient explanation, which is what a clinician actually wants at a
    bedside. "This patient's predicted risk is high mostly because of X and Y"
    is a usable sentence. A global feature-importance bar chart is not, and it
    is what most people show.

A29. FOUR LINKAGES, FOUR ANSWERS
    The comparison above is the answer to "which is right": none of them is,
    because the question is under-determined. Linkage defines what "distance
    between two clusters" means, and each definition encodes a different prior
    about cluster shape.

    Single linkage takes the MINIMUM distance between members. It chains: a
    sequence of near-neighbours drags distant points into one cluster, which is
    why its largest cluster swallows {single_share}% of the cohort while the
    rest fragment into near-singletons.

    Complete linkage takes the MAXIMUM, insisting every member be close to every
    other, and textbooks describe it as producing compact, roughly equal groups.
    It does not here -- its largest cluster still holds {complete_share}%. That
    is worth noticing rather than glossing: in {n_features} standardised
    dimensions almost every pair of patients is far apart and at similar
    distances, so the maximum-distance criterion has little to discriminate
    with. Average linkage behaves like single ({average_share}%).

    Only Ward's method produces a usable partition ({ward_sizes}), because it
    merges to minimise the increase in within-cluster variance rather than
    reasoning about any single pair.

    Now read the diagnostics carefully, because the naive reading is backwards.
    Cophenetic correlation is highest for {best_coph_method} ({best_coph:.2f}),
    and the best silhouette is {best_sil:.2f} -- which looks like a well
    separated solution until you notice both belong to the degenerate
    partitions. A silhouette is flattering when 99% of points sit in one cluster
    and the remainder are distant singletons: each point is close to its own
    cluster and far from the tiny others, and the score rewards that. Ward's
    partition -- the only one that divides the cohort at all -- scores
    {ward_sil:.2f}, which is honest and means no separation worth the name.

    A high internal validity score on a degenerate clustering is a trap, and
    quoting the best number across linkages without checking WHICH partition
    earned it is how people talk themselves into subtypes that are not there.

    What would make one of these a phenotype? Three things this analysis does
    not have. Stability -- the same groups recovered on a resample or a second
    cohort. External correlates -- groups that differ on something not used to
    build them, such as treatment response. And clinical coherence -- a
    cardiologist recognising the groups as patients they have met.

    This matters because the technique is not fanciful here: HFpEF phenogrouping
    by cluster analysis is real published cardiology. But those studies cluster
    on echocardiographic structure and function, and the defining variable --
    ejection fraction -- is exactly what this dataset lacks (04_clinical.py Q21).
    Clustering heart failure patients without it is phenotyping a condition with
    its defining measurement missing.

    So the honest framing is exploratory, and the honest conclusion is negative:
    the partition is unstable across linkage choices, weakly separated, and
    cannot be validated. Reporting it as "we identified four patient subtypes"
    would be the single most overstated sentence available in this project.

A30. THE CASE FOR THE TREE, AND AGAINST IT
    For. It is the only model here a clinician can apply without a computer:
    {n_leaves} terminal groups reached by three questions, with observed risks
    from {risk_min}% to {risk_max}% against a cohort rate of {prevalence}%. That
    is the form clinical risk tools actually take -- CHA2DS2-VASc, TIMI, Wells
    are all simple rules, and they are used precisely because they are simple.
    A model that is never applied has an effective AUC of nothing, and adoption
    is a real term in that equation rather than a consolation prize.

    Against. A single tree is unstable: refit it on a bootstrap resample and the
    split variables change, because each split is chosen greedily and every
    later split is conditional on it. It discards information by forcing
    continuous variables into thresholds -- creatinine does not become dangerous
    at one particular value, and 02_profile.py Q10 showed its risk curve rises
    then plateaus, which a threshold cannot express. And it is the worst
    DISCRIMINATING model in 05_modelling.py by a margin that, unlike every other
    pairwise comparison there, excludes zero: it is the one model genuinely
    beaten rather than merely different.

    One thing it is NOT is badly calibrated -- slope {tree_slope}, better than
    both the unpenalised regression and XGBoost. That is less impressive than it
    sounds. A tree can emit only as many distinct probabilities as it has
    leaves, {n_leaves} here, and each is the observed rate in its own training
    group; predictions that coarse have little room to be overconfident. Good
    calibration achieved by refusing to make fine distinctions is not the same
    virtue as good calibration across a continuous range.

    The resolution is not to choose. Report the penalised regression as the
    model, with odds ratios and a calibration curve, and offer the tree as a
    simplified companion with its performance cost stated explicitly. Presenting
    the tree ALONE would be indefensible; presenting it as a deliberately
    simplified companion, with the trade quantified, is a judgement a reviewer
    can evaluate.

    What would be indefensible either way is presenting the tree as though its
    thresholds were discovered facts about physiology. They are artefacts of a
    greedy algorithm on {n} patients, and a second sample would produce
    different numbers.
{rule}
"""


def main() -> None:
    viz.apply_style()
    configure_pandas()
    chf = analysis_frames().chf_train
    y = make_outcome(chf).values
    predictors = default_predictors(chf)

    header("SUPPORT2 -- interpretation")
    print(f"  CHF training cohort {len(chf):,}, {OUTCOME_LABEL}, "
          f"{int(y.sum()):,} events ({y.mean()*100:.1f}%)")
    print("  All fits on the training partition. The held-out 30% is not read.")

    pipe, Z, names = fit_xgb(chf, predictors, y)
    shap_r = compute_shap(pipe, Z, y)
    report_shap(shap_r, Z)

    Zs = design_matrix(chf, predictors, y)
    clust = compute_clustering(Zs, y)
    report_clustering(clust, y)

    tree_r = compute_tree(chf, predictors, y)
    report_tree(tree_r, y)

    header("FIGURES")
    for path in (figure_shap(shap_r, Z),
                 figure_dendrogram(clust, Zs),
                 figure_tree(tree_r)):
        print(f"  wrote {path.relative_to(Path(__file__).resolve().parent)}")

    # The tree's calibration slope is computed here rather than quoted from
    # 05_modelling.py, so A30 cannot drift out of step with the other script.
    tree_slope = calibration_metrics(y, cross_val_predictions(
        tree_r["pipe"], chf[predictors], y, n_repeats=CV_REPEATS,
        label="tree calibration"))["calibration_slope"]

    best_coph = max(clust.items(), key=lambda kv: kv[1]["cophenetic"])
    best_sil = max(c["silhouette"] for c in clust.values())
    facts = Facts(
        single_share=f"{clust['single']['largest_share']:.1f}",
        complete_share=f"{clust['complete']['largest_share']:.1f}",
        average_share=f"{clust['average']['largest_share']:.1f}",
        ward_share=f"{clust['ward']['largest_share']:.1f}",
        ward_sizes=", ".join(str(int(v)) for v in clust["ward"]["sizes"]),
        ward_sil=clust["ward"]["silhouette"],
        n_features=str(Zs.shape[1]),
        best_coph_method=best_coph[0],
        best_coph=best_coph[1]["cophenetic"],
        best_sil=best_sil,
        n_leaves=str(tree_r["n_leaves"]),
        tree_slope=f"{tree_slope:.2f}",
        risk_min=f"{tree_r['leaves'].risk.min():.1f}",
        risk_max=f"{tree_r['leaves'].risk.max():.1f}",
        prevalence=f"{y.mean()*100:.1f}",
        n=f"{len(chf):,}",
    )
    print(render_answers(ANSWERS, dict(facts, rule=RULE)))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        run_and_capture(main, OUT_DIR / "06_interpretation.txt")
