"""Shared data loading and feature engineering for the NACA airfoil explorer apps.

The raw dataset has one row per (airfoil, Reynolds number, angle of attack).
For clustering we collapse this to one row per airfoil at a fixed Reynolds
number, summarising each polar with a small set of geometry and performance
features.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "naca4series_24_airfoils.csv"

REQUIRED_COLUMNS = {
    "reynoldsNumber", "airfoilName", "angleOfAttack",
    "liftCoefficient", "dragCoefficient",
    "maxCamber", "maxCamberXloc", "maxThickness",
}

FEATURE_COLUMNS = [
    "maxCamber", "maxCamberXloc", "maxThickness",   # geometry
    "Cl_max", "Cd_min", "L_over_D_max", "AoA_at_LD_max",  # performance
]


def load_airfoil_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw NACA dataset and add the lift-to-drag ratio column.

    Raises FileNotFoundError / ValueError with a readable message so the
    calling app can display it.
    """
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at `{csv_path}`. "
            "See the README for how to obtain `naca4series_24_airfoils.csv`."
        )

    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}")

    df["L_over_D"] = df["liftCoefficient"] / df["dragCoefficient"].replace(0, np.nan)
    return df.sort_values(["reynoldsNumber", "airfoilName", "angleOfAttack"])


def summarise_at_reynolds(df: pd.DataFrame, reynolds: float) -> pd.DataFrame:
    """Collapse each airfoil's polar at one Reynolds number into a single row.

    Returns one row per airfoil with `airfoilName` plus FEATURE_COLUMNS,
    dropping airfoils with incomplete features.
    """
    df_re = df[df["reynoldsNumber"] == reynolds]
    if df_re.empty:
        return pd.DataFrame(columns=["airfoilName", *FEATURE_COLUMNS])

    summary = df_re.groupby("airfoilName", as_index=False).agg(
        maxCamber=("maxCamber", "first"),
        maxCamberXloc=("maxCamberXloc", "first"),
        maxThickness=("maxThickness", "first"),
        Cl_max=("liftCoefficient", "max"),
        Cd_min=("dragCoefficient", "min"),
        L_over_D_max=("L_over_D", "max"),
    )

    # Angle of attack at which each airfoil reaches its best L/D
    best_ld_idx = df_re.groupby("airfoilName")["L_over_D"].idxmax().dropna()
    best_ld = df_re.loc[best_ld_idx, ["airfoilName", "angleOfAttack"]].rename(
        columns={"angleOfAttack": "AoA_at_LD_max"}
    )

    summary = summary.merge(best_ld, on="airfoilName", how="left")
    return summary.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
