"""Interactive DBSCAN + UMAP explorer for NACA airfoils at a fixed Reynolds number.

Run with:  streamlit run src/dbscan_umap_app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st
import umap
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from airfoil_features import FEATURE_COLUMNS, load_airfoil_data, summarise_at_reynolds

st.set_page_config(page_title="Airfoil Clustering Explorer: DBSCAN + UMAP", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_airfoil_data()


@st.cache_data
def umap_embedding(X_scaled):
    # Cached so moving the DBSCAN sliders doesn't re-fit UMAP
    return umap.UMAP(n_components=2, random_state=42).fit_transform(X_scaled)


try:
    df = get_data()
except (FileNotFoundError, ValueError) as err:
    st.error(str(err))
    st.stop()

st.title("Airfoil Clustering Explorer: DBSCAN + UMAP")
st.caption(
    "Density-based clustering (DBSCAN) runs on the standardised 7-feature space; UMAP is "
    "used only to visualise the result in 2D. Points labelled `-1` are noise."
)

col1, col2, col3 = st.columns(3)
with col1:
    reynolds = st.selectbox(
        "Reynolds number", sorted(df["reynoldsNumber"].unique()), format_func=lambda r: f"{r:,}"
    )
with col2:
    eps = st.slider("DBSCAN eps", min_value=0.1, max_value=3.0, value=1.0, step=0.1)
with col3:
    min_samples = st.slider("DBSCAN min_samples", min_value=2, max_value=10, value=3)

summary = summarise_at_reynolds(df, reynolds)
if len(summary) < min_samples:
    st.warning(f"Only {len(summary)} airfoils available; set min_samples to ≤ {len(summary)}.")
    st.stop()

X_scaled = StandardScaler().fit_transform(summary[FEATURE_COLUMNS])
labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
summary["cluster"] = labels.astype(str)

n_clusters = len(set(labels) - {-1})
n_noise = int((labels == -1).sum())

colA, colB, colC = st.columns(3)
colA.metric("Clusters found", n_clusters)
colB.metric("Noise points", n_noise)

# Silhouette is only defined with ≥ 2 non-noise clusters
non_noise = labels != -1
if len(set(labels[non_noise])) >= 2:
    colC.metric(
        "Silhouette (excl. noise)",
        f"{silhouette_score(X_scaled[non_noise], labels[non_noise]):.3f}",
    )
else:
    colC.metric("Silhouette (excl. noise)", "n/a")

X_umap = umap_embedding(X_scaled)
plot_df = pd.DataFrame({
    "UMAP1": X_umap[:, 0],
    "UMAP2": X_umap[:, 1],
    "cluster": summary["cluster"],
    "airfoil": summary["airfoilName"],
    "L_over_D_max": summary["L_over_D_max"].round(2),
})

fig = px.scatter(
    plot_df, x="UMAP1", y="UMAP2", color="cluster",
    hover_data=["airfoil", "L_over_D_max"],
    title=f"UMAP projection  |  Re = {reynolds:,}  |  eps = {eps}  |  min_samples = {min_samples}",
)
st.plotly_chart(fig, width="stretch")

st.subheader("Cluster mean profiles")
clustered = summary[summary["cluster"] != "-1"]
if clustered.empty:
    st.info("DBSCAN found no clusters with these parameters. Try increasing eps.")
else:
    st.dataframe(clustered.groupby("cluster")[FEATURE_COLUMNS].mean().round(4), width="stretch")

st.subheader("Airfoils per cluster (sorted by max L/D)")
st.dataframe(
    summary.sort_values(["cluster", "L_over_D_max"], ascending=[True, False]),
    width="stretch",
    hide_index=True,
)
