"""Stage 1: monsoon risk regions by clustering the 77 districts.

Each district is described by its 2017 to 2022 profile: the monthly rate of damaging days for each
hazard (4 hazards x 7 months), the monthly ERA5 rain climatology (7) and two terrain numbers. The
profile is standardised and reduced by PCA, then clustered with K-means, Ward and a Gaussian mixture.
The profile uses same-day rates of the training seasons, so it describes history, not the target.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config as C

HAZ = list(C.HAZARDS.values())
MONTHS = list(range(4, 11))
TERRAIN = ["elev_mean", "elev_relief"]


def district_profile(F, seasons):
    """One row per district: 28 hazard-month damaging-day rates, 7 monthly mean daily rain values, 2 terrain values."""
    tr = F[F["season"].isin(seasons)]
    parts = []
    for h in HAZ:
        r = (
            tr.groupby(["district", "month"])[f"day_{h}"]
            .mean()
            .unstack("month")
            .reindex(columns=MONTHS)
        )
        r.columns = [f"rate_{h}_m{m}" for m in MONTHS]
        parts.append(r)
    rain = (
        tr.groupby(["district", "month"])["rain_1d"].mean().unstack("month").reindex(columns=MONTHS)
    )
    rain.columns = [f"rain_m{m}" for m in MONTHS]
    parts.append(rain)
    parts.append(tr.groupby("district")[TERRAIN].first())
    return pd.concat(parts, axis=1).sort_index()


def reduce(profile, variance=0.90, seed=C.SEED):
    """Standardise and PCA to the given share of variance. Returns Z, scaler and pca."""
    scaler = StandardScaler().fit(profile)
    pca = PCA(n_components=variance, svd_solver="full", random_state=seed).fit(
        scaler.transform(profile)
    )
    return pca.transform(scaler.transform(profile)), scaler, pca


def fit_clusterer(method, Z, k, seed=C.SEED):
    """Fit one clusterer and return (fitted object or None, labels)."""
    if method == "kmeans":
        km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(Z)
        return km, km.labels_
    if method == "ward":
        return None, AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    if method == "gmm":
        gm = GaussianMixture(
            n_components=k, covariance_type="diag", n_init=10, random_state=seed, reg_covar=1e-3
        ).fit(Z)
        return gm, gm.predict(Z)
    raise ValueError(method)


def validity(Z, labels):
    """Silhouette, Davies-Bouldin and Calinski-Harabasz for one labelling."""
    return {
        "silhouette": silhouette_score(Z, labels),
        "davies_bouldin": davies_bouldin_score(Z, labels),
        "calinski_harabasz": calinski_harabasz_score(Z, labels),
    }


def validity_table(Z, ks=range(2, 9), seed=C.SEED):
    """Validity indices for every method and k, plus BIC for the mixture."""
    rows = []
    for method in ("kmeans", "ward", "gmm"):
        for k in ks:
            obj, lab = fit_clusterer(method, Z, k, seed)
            r = {
                "method": method,
                "k": k,
                **validity(Z, lab),
                "smallest_cluster": int(np.bincount(lab).min()),
            }
            if method == "gmm":
                r["bic"] = obj.bic(Z)
            rows.append(r)
    return pd.DataFrame(rows)


def bootstrap_ari(F, seasons, method, k, reference, n_boot=200, seed=C.SEED):
    """Stability: resample the training seasons with replacement, rebuild the profile, refit, ARI against `reference`."""
    rng = np.random.default_rng(seed)
    per_season = {s: district_profile(F, [s]) for s in seasons}
    out = []
    for b in range(n_boot):
        pick = rng.choice(seasons, size=len(seasons), replace=True)
        prof = sum(per_season[s] for s in pick) / len(pick)
        Z, _, _ = reduce(prof, seed=seed + b)
        _, lab = fit_clusterer(method, Z, k, seed=seed + b)
        out.append(adjusted_rand_score(reference, lab))
    return np.asarray(out)


def belts(T):
    """Ecological belt (Terai, Hill, Mountain) proxy from the ward-elevation summary of each district.

    Mountain: mean ward elevation of at least 1,700 m and at least 5% of wards above 3,000 m.
    Terai: mean ward elevation below 700 m and fewer than 15% of wards above 1,000 m (the plain,
    the Siwalik foothills and the inner Terai valleys). Hill: everything else. This is a proxy
    derived from elevation, not the official belt map.
    """
    T = T.set_index("district") if "district" in T else T
    belt = pd.Series("Hill", index=T.index)
    belt[(T["elev_mean"] >= 1700) & (T["share_above_3000m"] >= 0.05)] = "Mountain"
    belt[(T["elev_mean"] < 700) & (T["share_above_1000m"] < 0.15)] = "Terai"
    return belt


def region_encodings(Z, districts, km, gm):
    """Per district: hard K-means label, GMM posterior probabilities and distances to the K-means centroids."""
    out = pd.DataFrame({"district": districts, "region": km.predict(Z)})
    post = gm.predict_proba(Z)
    for j in range(post.shape[1]):
        out[f"region_post_{j}"] = post[:, j]
    dist = km.transform(Z)
    for j in range(dist.shape[1]):
        out[f"region_dist_{j}"] = dist[:, j]
    return out
