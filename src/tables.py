"""Write the paper's LaTeX tables from the CSV files in results/.

    uv run python -m src.tables

Every number in the paper's tables comes from here, so the tables cannot drift from the results.
Output goes to paper/tables/*.tex and is \\input by paper/main.tex.
"""

import json

import pandas as pd

from . import config as C
from . import features as Fe
from . import models as M

OUT = C.ROOT / "paper" / "tables"
SHORT = {
    "climatology": "Clim.",
    "persistence": "Persist.",
    "dahal_threshold": "D\\&H thr.",
    "refit_threshold": "ERA5 thr.",
    "year_only_control": "Year ctrl.",
    "lr": "LR",
    "rf": "RF",
    "hgb": "HGB",
    "svm_linear": "SVM-L",
    "svm_rbf": "SVM-RBF",
}
NAMES = {
    "climatology": "Climatology (district $\\times$ month)",
    "persistence": "Persistence (known at $t$)",
    "dahal_threshold": "Dahal and Hasegawa threshold",
    "refit_threshold": "Refitted ERA5 percentile threshold",
    "year_only_control": "Year-only control",
    "lr": "Logistic regression",
    "rf": "Random forest",
    "hgb": "Gradient boosting (HGB)",
    "svm_linear": "SVM, linear",
    "svm_rbf": "SVM, RBF (subsample)",
}
FEATURE_NOTES = {
    "rain_1d": "ERA5 precipitation on day $t$ (mm)",
    "rain_3d": "Sum over $t-2$ to $t$ (mm)",
    "rain_7d": "Sum over 7 days (mm)",
    "rain_14d": "Sum over 14 days (mm)",
    "rain_30d": "Sum over 30 days (mm)",
    "rain_api": "Antecedent precipitation index, decay 0.9",
    "rain_hours_1d": "Hours with precipitation on day $t$",
    "tmax_1d": "Maximum 2 m temperature on day $t$ (\\textdegree C)",
    "rain_3d_pct": "Percentile of the 3-day sum in the district's 2017 to 2022 distribution",
    "doy_sin": "Day of year, sine",
    "doy_cos": "Day of year, cosine",
    "month": "Calendar month",
    "elev_mean": "Mean ward elevation (m)",
    "elev_relief": "Ward elevation range (m)",
    "share_above_1000m": "Share of wards above 1,000 m",
    "share_above_3000m": "Share of wards above 3,000 m",
    "recent_dam_3d": "Damaging incidents known at $t$, dated $t-3$ to $t$",
    "recent_dam_7d": "Same, $t-7$ to $t$",
    "nbr_recent_dam_3d": "Same as recent\\_dam\\_3d, summed over neighbouring districts",
    "nbr_recent_dam_7d": "Same, 7 days",
    "recent_flood_7d": "Flood incidents, 7 days",
    "recent_landslide_7d": "Landslide incidents, 7 days",
    "recent_heavy_rain_7d": "Heavy-rainfall incidents, 7 days",
    "recent_lightning_7d": "Lightning incidents, 7 days",
    "te_dam": "District $\\times$ month rate of damaging days (shrunk, leave-one-season-out)",
    "te_flood": "Same, flood",
    "te_landslide": "Same, landslide",
    "te_heavy_rain": "Same, heavy rainfall",
    "te_lightning": "Same, lightning",
    "future_rain_3d": "Observed ERA5 rain, $t+1$ to $t+3$ (mm); set B only",
    "future_rain_max_1d": "Maximum daily rain, $t+1$ to $t+3$; set B only",
    "future_rain_hours_3d": "Rain hours, $t+1$ to $t+3$; set B only",
}


def ci(df, col, digits=3):
    return [
        f"{e:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
        for e, lo, hi in zip(df[col], df[f"{col}_lo"], df[f"{col}_hi"])
    ]


def write(name, body):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(body)


def table_splits():
    b = pd.read_csv(C.RESULTS / "eda-class-balance.csv", index_col=0)
    short = {
        "train": "2017--2022",
        "validate": "2023",
        "test": "2024--2025",
        "check": "2026 (to 16 Aug)",
    }
    rows = [
        f"{s.capitalize() if s != 'check' else 'Check'} & {short[s]} & {b.loc[s, 'rows']:,} & {b.loc[s, 'positives']:,} & {b.loc[s, 'rate']:.2f} & {b.loc[s, 'y_rain_rate']:.2f} \\\\"
        for s in ["train", "validate", "test", "check"]
    ]
    write(
        "splits",
        "\\begin{tabular}{llrrrr}\n\\toprule\nSplit & Seasons & Rows & Positives & $y$ (\\%) & $y_{\\mathrm{rain}}$ (\\%) \\\\\n\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n\\end{tabular}\n",
    )


def table_cluster():
    v = pd.read_csv(C.RESULTS / "cluster-validity.csv")
    v = v[v["space"].str.startswith("4 ")]
    st = pd.read_csv(C.RESULTS / "cluster-stability.csv").set_index("method")
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Method & $k$ & Silhouette & DB & CH & Min. size \\\\",
        "\\midrule",
    ]
    label = {"kmeans": "K-means", "ward": "Ward", "gmm": "GMM"}
    for m in ("kmeans", "ward", "gmm"):
        d = v[v["method"] == m]
        for i, r in d.iterrows():
            lead = label[m] if r["k"] == 2 else ""
            k = f"\\textbf{{{int(r['k'])}}}" if int(r["k"]) == 6 else f"{int(r['k'])}"
            lines.append(
                f"{lead} & {k} & {r['silhouette']:.3f} & {r['davies_bouldin']:.3f} & {r['calinski_harabasz']:.1f} & {int(r['smallest_cluster'])} \\\\"
            )
        if m != "gmm":
            lines.append("\\addlinespace[2pt]")
    lines += [
        "\\midrule",
        "\\multicolumn{6}{l}{At $k=6$: bootstrap ARI [5th, 95th]; ARI vs belts / provinces} \\\\",
    ]
    for m in ("kmeans", "ward", "gmm"):
        r = st.loc[m]
        lines.append(
            f"{label[m]} & 6 & \\multicolumn{{2}}{{l}}{{{r['boot_ari_mean']:.2f} [{r['boot_ari_q05']:.2f}, {r['boot_ari_q95']:.2f}]}} & {r['ari_vs_belts']:.2f} / {r['ari_vs_provinces']:.2f} & \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("cluster-validity", "\n".join(lines) + "\n")


def table_main():
    """Two single-column tables: (i) ranking skill with intervals, (ii) probabilistic and operational metrics."""
    b = pd.read_csv(C.RESULTS / "test-baselines.csv", index_col=0)
    m = pd.read_csv(C.RESULTS / "test-main-set-a.csv", index_col=0)
    fs = pd.read_csv(C.RESULTS / "test-feature-sets.csv", index_col=0)
    cal = pd.read_csv(C.RESULTS / "calibration-sensitivity.csv")
    pooled = cal[cal["calibration"].str.startswith("pooled")].set_index("model")["bss"]
    base = ["climatology", "persistence", "dahal_threshold", "refit_threshold", "year_only_control"]
    models = ["lr", "rf", "hgb", "svm_linear", "svm_rbf"]
    # (i) ranking
    li = [
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Model & PR-AUC [95\\% CI] & $\\Delta$ vs Clim. [95\\% CI] \\\\",
        "\\midrule",
        "\\multicolumn{3}{l}{\\textit{Baselines}} \\\\",
    ]
    for k in base:
        r = b.loc[k]
        delta = (
            ""
            if k == "climatology"
            else f"{r['pr_auc_diff_vs_ref']:+.3f} [{r['pr_auc_diff_lo']:+.3f}, {r['pr_auc_diff_hi']:+.3f}]"
        )
        li.append(f"{SHORT[k]} & {ci(b.loc[[k]], 'pr_auc')[0]} & {delta} \\\\")
    li.append("\\midrule\\multicolumn{3}{l}{\\textit{Models, set A (known at $t$)}} \\\\")
    for k in models:
        r = m.loc[k]
        li.append(
            f"{SHORT[k]} & {ci(m.loc[[k]], 'pr_auc')[0]} & {r['pr_auc_diff_vs_ref']:+.3f} [{r['pr_auc_diff_lo']:+.3f}, {r['pr_auc_diff_hi']:+.3f}] \\\\"
        )
    li.append("\\midrule\\multicolumn{3}{l}{\\textit{RF floor and ceiling}} \\\\")
    for k, lab in (("rf_NOWX", "RF, no weather"), ("rf_B", "RF, set B")):
        r = fs.loc[k]
        li.append(
            f"{lab} & {ci(fs.loc[[k]], 'pr_auc')[0]} & {r['pr_auc_diff_vs_ref']:+.3f} [{r['pr_auc_diff_lo']:+.3f}, {r['pr_auc_diff_hi']:+.3f}] \\\\"
        )
    li += ["\\bottomrule", "\\end{tabular}"]
    write("main-results", "\n".join(li) + "\n")
    # (ii) probabilistic and operational; POD, FAR and CSI at the 5% budget with random tie-breaking
    al = pd.read_csv(C.RESULTS / "alert-budget.csv", index_col=0)
    li = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Model & BSS (2023) & BSS (pooled) & POD & FAR & CSI \\\\",
        "\\midrule",
    ]
    for k in ("climatology", "year_only_control"):
        r = al.loc[k]
        li.append(
            f"{SHORT[k]} & {b.loc[k, 'bss']:.3f} & & {r['pod']:.3f} & {r['far']:.3f} & {r['csi']:.3f} \\\\"
        )
    li.append("\\midrule")
    for k in models:
        r = al.loc[k]
        li.append(
            f"{SHORT[k]} & {m.loc[k, 'bss']:.3f} & {pooled[k]:.3f} & {r['pod']:.3f} & {r['far']:.3f} & {r['csi']:.3f} \\\\"
        )
    li.append("\\midrule")
    for k, lab in (("rf_NOWX", "RF, no weather"), ("rf_B", "RF, set B")):
        r = al.loc[k]
        li.append(
            f"{lab} & {fs.loc[k, 'bss']:.3f} & & {r['pod']:.3f} & {r['far']:.3f} & {r['csi']:.3f} \\\\"
        )
    li += ["\\bottomrule", "\\end{tabular}"]
    write("main-results-ops", "\n".join(li) + "\n")


def table_rules():
    """Each rule baseline at its own alert rate against climatology and models at the same rate."""
    r = pd.read_csv(C.RESULTS / "alert-rules-own-rate.csv")
    scorers = ["rule", "climatology", "lr", "rf"]
    head = {"rule": "Rule", "climatology": "Clim.", "lr": "LR", "rf": "RF"}
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Rule (alert rate) & & " + " & ".join(head[k] for k in scorers) + " \\\\",
        "\\midrule",
    ]
    for i, rule in enumerate(["persistence", "dahal_threshold", "refit_threshold"]):
        d = r[r["rule"] == rule].set_index("scorer")
        rate = d.loc["rule", "alert_rate"]
        for j, met in enumerate(("pod", "far", "csi")):
            lead = f"{SHORT[rule]} ({100 * rate:.1f}\\%)" if j == 0 else ""
            cells = " & ".join(f"{d.loc[k, met]:.3f}" for k in scorers)
            lines.append(f"{lead} & {met.upper()} & {cells} \\\\")
        if i < 2:
            lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("alert-rules", "\n".join(lines) + "\n")


def table_feature_sets():
    """PR-AUC per feature set and the paired differences A minus NOWX and B minus A."""
    a = pd.read_csv(C.RESULTS / "ablation-paired.csv", index_col=0)

    def d(r, c):
        return f"{r[c]:+.3f} [{r[c + '_lo']:+.3f}, {r[c + '_hi']:+.3f}]"

    lines = [
        "\\begin{tabular}{lrrrll}",
        "\\toprule",
        "Model & NOWX & A & B & A $-$ NOWX [95\\% CI] & B $-$ A [95\\% CI] \\\\",
        "\\midrule",
    ]
    for k in ["lr", "rf", "hgb", "svm_linear", "svm_rbf"]:
        r = a.loc[k]
        lines.append(
            f"{SHORT[k]} & {r['pr_auc_nowx']:.3f} & {r['pr_auc_a']:.3f} & {r['pr_auc_b']:.3f} & {d(r, 'a_minus_nowx')} & {d(r, 'b_minus_a')} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("feature-sets", "\n".join(lines) + "\n")


def table_regions():
    r = pd.read_csv(C.RESULTS / "test-region-encodings.csv", index_col=0)
    enc_names = {
        "none": "none (set A)",
        "label": "+ K-means label",
        "posterior": "+ GMM posteriors",
        "distance": "+ centroid distances",
        "district": "+ district one-hot",
    }
    lines = [
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Model & Encoding & PR-AUC & $\\Delta$ vs none [95\\% CI] \\\\",
        "\\midrule",
    ]
    for m in [x for x in ["rf", "lr"] if f"{x}_none" in r.index]:
        for i, (e, lab) in enumerate(enc_names.items()):
            x = r.loc[f"{m}_{e}"]
            d = (
                ""
                if e == "none"
                else f"{x['pr_auc_diff_vs_none']:+.3f} [{x['pr_auc_diff_vs_none_lo']:+.3f}, {x['pr_auc_diff_vs_none_hi']:+.3f}]"
            )
            lines.append(f"{SHORT[m] if i == 0 else ''} & {lab} & {x['pr_auc']:.3f} & {d} \\\\")
        lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("region-encodings", "\n".join(lines) + "\n")


def table_seasons():
    s = pd.read_csv(C.RESULTS / "skill-per-season.csv")
    rain = pd.read_csv(C.RESULTS / "test-rain-target.csv", index_col=0)
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Season & Role & Base & Clim. & RF & LR & RF BSS \\\\",
        "\\midrule",
    ]
    for season in [2023, 2024, 2025, 2026]:
        rf = s[(s["model"] == "rf") & (s["season"] == season)].iloc[0]
        lr = s[(s["model"] == "lr") & (s["season"] == season)].iloc[0]
        role = {2023: "calib.", 2024: "test", 2025: "test", 2026: "check"}[season]
        lines.append(
            f"{season} & {role} & {rf['base_rate']:.3f} & {rf['clim_pr_auc']:.3f} & {rf['pr_auc']:.3f} & {lr['pr_auc']:.3f} & {rf['bss']:+.3f} \\\\"
        )
    lines.append(
        "\\midrule\\multicolumn{7}{l}{\\textit{Rain-driven target, test 2024 to 2025}} \\\\"
    )
    lines.append(
        f"2024--25 & test & {rain.loc['climatology', 'base_rate']:.3f} & {rain.loc['climatology', 'pr_auc']:.3f} & {rain.loc['rf_A', 'pr_auc']:.3f} & {rain.loc['lr_A', 'pr_auc']:.3f} & {rain.loc['rf_A', 'bss']:+.3f} \\\\"
    )
    lines.append(
        f"2024--25 & set B & & & {rain.loc['rf_B', 'pr_auc']:.3f} & {rain.loc['lr_B', 'pr_auc']:.3f} & {rain.loc['rf_B', 'bss']:+.3f} \\\\"
    )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("per-season", "\n".join(lines) + "\n")


def table_errors():
    if not (C.RESULTS / "errors-per-region.csv").exists():
        return
    reg = pd.read_csv(C.RESULTS / "errors-per-region.csv")
    belt = pd.read_csv(C.RESULTS / "errors-per-belt.csv")
    haz = pd.read_csv(C.RESULTS / "errors-per-hazard.csv")
    lines = [
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Stratum & Pos. & Base & Clim. & RF & POD \\\\",
        "\\midrule",
    ]
    for _, r in reg.iterrows():
        lines.append(
            f"Region {int(r['region'])} & {int(r['positives'])} & {r['base_rate']:.3f} & {r['clim_pr_auc']:.3f} & {r['pr_auc']:.3f} & {r['pod_at_national_budget']:.3f} \\\\"
        )
    lines.append("\\addlinespace[2pt]")
    for _, r in belt.iterrows():
        lines.append(
            f"{r['belt']} & {int(r['positives'])} & {r['base_rate']:.3f} & {r['clim_pr_auc']:.3f} & {r['pr_auc']:.3f} & {r['pod_at_national_budget']:.3f} \\\\"
        )
    lines.append("\\addlinespace[2pt]")
    hn = {
        "flood": "Flood",
        "landslide": "Landslide",
        "heavy_rain": "Heavy rainfall",
        "lightning": "Lightning",
    }
    for _, r in haz.iterrows():
        lines.append(
            f"{hn[r['hazard']]} (own label) & {int(r['positives'])} & {r['base_rate']:.3f} & {r['pr_auc_climatology']:.3f} & {r['pr_auc_pooled_score']:.3f} & {r['pod_at_national_budget']:.3f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("errors", "\n".join(lines) + "\n")


def table_hyperparameters():
    best = pd.read_csv(C.RESULTS / "best-params.csv").set_index("model")
    # Mean of the raw fold scores: best-params.csv stores it rounded to 4 decimals.
    fold_mean = chosen_folds().mean()
    lines = [
        "\\begin{tabular}{p{0.8cm}p{3.0cm}p{1.95cm}r}",
        "\\toprule",
        "Model & Grid and fixed settings & Chosen & Fold PR-AUC \\\\",
        "\\midrule",
    ]
    fixed = {
        "lr": "L2, class weight balanced, standardised inputs",
        "rf": "300 trees, class weight balanced-subsample",
        "hgb": "300 iterations, min leaf 50, L2 1.0, class weight balanced",
        "svm_linear": "LinearSVC, class weight balanced, standardised",
        "svm_rbf": "RBF, $\\gamma$ = scale, 12,000-row stratified subsample",
    }
    for k in ["lr", "rf", "hgb", "svm_linear", "svm_rbf"]:
        g = ";\\newline ".join(
            f"{p} $\\in$ \\{{{', '.join(str(v) for v in vals)}\\}}"
            for p, vals in M.GRIDS[k].items()
        ).replace("_", "\\_")
        chosen = ", ".join(
            f"{p} = {v}" for p, v in json.loads(best.loc[k, "params"]).items()
        ).replace("_", "\\_")
        lines.append(f"{SHORT[k]} & {g};\\newline {fixed[k]} & {chosen} & {fold_mean[k]:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("hyperparameters", "\n".join(lines) + "\n")


def chosen_folds():
    """Fold PR-AUC (rows: held-out season) of each model's chosen setting, from tuning-folds.csv."""
    folds = pd.read_csv(C.RESULTS / "tuning-folds.csv")
    best = pd.read_csv(C.RESULTS / "best-params.csv").set_index("model")
    cols = {}
    for k in ["lr", "rf", "hgb", "svm_linear", "svm_rbf"]:
        d = folds[folds["model"] == k]
        for p, v in json.loads(best.loc[k, "params"]).items():
            d = d[d[p] == v] if isinstance(v, str) else d[pd.to_numeric(d[p]) == float(v)]
        s = d.set_index("held_out_season")["pr_auc"]
        assert len(s) == 6 and abs(s.mean() - best.loc[k, "loso_mean_pr_auc"]) < 1e-4, k
        cols[k] = s
    return pd.DataFrame(cols)


def table_tuning_folds():
    """Leave-one-season-out PR-AUC of each model's chosen setting, per held-out season."""
    t = chosen_folds()
    rates = pd.read_csv(C.RESULTS / "eda-rates-per-year.csv", index_col=0)["y_rate"]
    models = list(t.columns)
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Held out & Base (\\%) & " + " & ".join(SHORT[k] for k in models) + " \\\\",
        "\\midrule",
    ]
    for season, r in t.iterrows():
        cells = " & ".join(f"{r[k]:.3f}" for k in models)
        lines.append(f"{season} & {rates.loc[season]:.1f} & {cells} \\\\")
    lines.append("\\midrule")
    for lab, stat in (("Mean", t.mean()), ("SD", t.std()), ("Min", t.min()), ("Max", t.max())):
        lines.append(f"{lab} & & " + " & ".join(f"{stat[k]:.3f}" for k in models) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("tuning-folds", "\n".join(lines) + "\n")


def table_budget_counts():
    """Alert outcomes at the 5% budget on test (random tie-breaking), recovered from n, positives, POD and alert rate."""
    al = pd.read_csv(C.RESULTS / "alert-budget.csv", index_col=0)
    rows = [
        (SHORT[k], al.loc[k]) for k in ["climatology", "lr", "rf", "hgb", "svm_linear", "svm_rbf"]
    ]
    rows.append(("RF, set B", al.loc["rf_B"]))
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & Hits & False alarms & Misses & Correct neg. \\\\",
        "\\midrule",
    ]
    for lab, r in rows:
        n, pos = int(r["n"]), int(r["positives"])
        alerts, hits = round(r["alert_rate"] * n), round(r["pod"] * pos)
        fa = alerts - hits
        assert abs(r["pod"] * pos - hits) < 1e-6 and abs(r["far"] * alerts - fa) < 1e-6, lab
        lines.append(
            f"{lab} & {hits:,} & {fa:,} & {pos - hits:,} & {n - alerts - (pos - hits):,} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("budget-counts", "\n".join(lines) + "\n")


def table_run_times():
    """The clean run of the five notebooks: the first five lines of results/notebook-run-times.txt."""
    lines = [
        "\\begin{tabular}{lrrl}",
        "\\toprule",
        "Notebook & Exit & Seconds & Finished \\\\",
        "\\midrule",
    ]
    for raw in (C.RESULTS / "notebook-run-times.txt").read_text().splitlines()[:5]:
        name, *fields = raw.split()
        f = dict(x.split("=", 1) for x in fields[:3])
        when = f"{f['finished']} {fields[3]}"
        lines.append(f"{name} & {f['exit']} & {f['seconds']} & {when} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("run-times", "\n".join(lines) + "\n")


def table_features():
    lines = [
        "\\begin{tabular}{p{0.9cm}p{3.0cm}p{3.55cm}}",
        "\\toprule",
        "Group & Feature & Definition \\\\",
        "\\midrule",
    ]
    groups = [
        ("Rain (A)", Fe.RAIN_A),
        ("Season", Fe.SEASON),
        ("Terrain", Fe.TERRAIN),
        ("Recent incidents", Fe.RECENT),
        ("History", Fe.HISTORY),
        ("Future rain (B)", Fe.FUTURE),
    ]
    for g, cols in groups:
        for i, c in enumerate(cols):
            lines.append(
                f"{g if i == 0 else ''} & \\texttt{{{c.replace('_', chr(92) + '_')}}} & {FEATURE_NOTES[c]} \\\\"
            )
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("features", "\n".join(lines) + "\n")


def main():
    for f in (
        table_splits,
        table_cluster,
        table_main,
        table_rules,
        table_feature_sets,
        table_regions,
        table_seasons,
        table_errors,
        table_hyperparameters,
        table_tuning_folds,
        table_budget_counts,
        table_features,
        table_run_times,
    ):
        f()
    print("tables written to", OUT, sorted(p.name for p in OUT.glob("*.tex")))


if __name__ == "__main__":
    main()
