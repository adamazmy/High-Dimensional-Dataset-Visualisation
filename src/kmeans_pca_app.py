"""Interactive KMeans + PCA explorer for NACA airfoils at a fixed Reynolds number.

Run with:  streamlit run src/kmeans_pca_app.py
"""

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from airfoil_features import FEATURE_COLUMNS, load_airfoil_data, summarise_at_reynolds

st.set_page_config(page_title="Airfoil Clustering Explorer: KMeans + PCA", layout="wide")


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_airfoil_data()


try:
    df = get_data()
except (FileNotFoundError, ValueError) as err:
    st.error(str(err))
    st.stop()

st.title("Airfoil Clustering Explorer")
st.caption(
    "Each point is one NACA 4-series airfoil, summarised by 3 geometry and 4 performance "
    "features at the selected Reynolds number. Features are standardised, clustered with "
    "KMeans, and projected to 2D with PCA for display."
)

col1, col2 = st.columns(2)
with col1:
    reynolds = st.selectbox(
        "Reynolds number", sorted(df["reynoldsNumber"].unique()), format_func=lambda r: f"{r:,}"
    )
with col2:
    k = st.slider("Number of clusters (k)", min_value=2, max_value=8, value=4)

summary = summarise_at_reynolds(df, reynolds)
if len(summary) <= k:
    st.warning(f"Only {len(summary)} airfoils available at this Reynolds number; reduce k.")
    st.stop()

X_scaled = StandardScaler().fit_transform(summary[FEATURE_COLUMNS])
labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(X_scaled)
summary["cluster"] = labels.astype(str)  # string -> discrete Plotly colours

st.metric("Silhouette score", f"{silhouette_score(X_scaled, labels):.3f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
var_explained = pca.explained_variance_ratio_ * 100

plot_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "cluster": summary["cluster"],
    "airfoil": summary["airfoilName"],
    "L_over_D_max": summary["L_over_D_max"].round(2),
})

fig = px.scatter(
    plot_df, x="PC1", y="PC2", color="cluster",
    hover_data=["airfoil", "L_over_D_max"],
    labels={
        "PC1": f"PC1 ({var_explained[0]:.1f}% variance)",
        "PC2": f"PC2 ({var_explained[1]:.1f}% variance)",
    },
    title=f"PCA projection  |  Re = {reynolds:,}  |  k = {k}",
)
st.plotly_chart(fig, width="stretch")

st.subheader("Cluster mean profiles")
st.dataframe(summary.groupby("cluster")[FEATURE_COLUMNS].mean().round(4), width="stretch")

st.subheader("Airfoils per cluster (sorted by max L/D)")
st.dataframe(
    summary.sort_values(["cluster", "L_over_D_max"], ascending=[True, False]),
    width="stretch",
    hide_index=True,
)
