"""Sample size summary — demographics breakdown by task, sex, and proband status."""

import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def imports():
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    return mo, pd, Path


@app.cell
def parameters():
    # Parameters — adjust for your environment
    # Path to the pipeline tracking output directory containing
    # Data_tracking_<task>_detailed_<date>.csv files
    tracking_dir = ""  # e.g. "/path/to/tracking/output_dfs/pipeline_outputs/2025_12_19"
    return (tracking_dir,)


@app.cell
def load_data(pd, Path, tracking_dir):
    """Load tracking CSVs for all tasks and combine."""
    if not tracking_dir:
        print("Set tracking_dir parameter to the pipeline tracking output directory.")
        return pd.DataFrame(), []

    path_date = Path(tracking_dir)
    if not path_date.exists():
        print(f"Directory not found: {path_date}")
        return pd.DataFrame(), []

    tasks = [f.name.split("_")[2] for f in path_date.glob("*.csv")]
    print(f"Found tasks: {tasks}")

    dfs = []
    for task in tasks:
        csv_files = list(path_date.glob(f"Data_tracking_{task}_detailed_*.csv"))
        if not csv_files:
            continue
        path_file = csv_files[0]
        df = pd.read_csv(path_file)
        df["task"] = task
        df = df[df["EEG Raw Files"] == "Yes"]
        dfs.append(df)

    if not dfs:
        print("No tracking CSVs found.")
        return pd.DataFrame(), tasks

    df = pd.concat(dfs)
    df["is_proband"] = [str(id_)[-1] == "P" for id_ in df["bids_id"].values]
    df["id_nb"] = [str(id_)[:4] for id_ in df["bids_id"].values]
    df = df.rename(columns={"eeg_age": "age"})

    return df, tasks


@app.cell
def raw_counts(mo, pd, df):
    """Show raw sample counts per task."""
    if df.empty:
        return ()
    counts = df.groupby("task").count()["sex"]
    mo.md("## Raw sample counts per task")
    mo.ui.table(pd.DataFrame(counts).rename(columns={"sex": "N"}))
    return ()


@app.cell
def demographics_table(mo, pd, df):
    """Build demographics breakdown table (probands vs relatives, by sex)."""
    if df.empty:
        return ()

    groups = df.groupby(["task", "is_proband", "sex"])

    # Count N
    N = groups.count().pivot_table(
        index=["task", "sex"], values="site", columns="is_proband",
    )
    N = N.fillna(0).astype(int)
    N.columns = [["relatives", "probands"], ["N", "N"]]

    # Mean age
    mean = groups.mean(numeric_only=True)
    mean["age"] = mean["age"].map("{:.1f}".format)
    mean = mean.reset_index().pivot(
        index=["task", "sex"], values="age", columns="is_proband",
    )

    # SD age
    sd = groups.std(numeric_only=True)
    sd["age"] = sd["age"].map("{:.1f}".format)
    sd = sd.reset_index().pivot(
        index=["task", "sex"], values="age", columns="is_proband",
    )

    display_df = mean + " \u00b1" + sd
    display_df.columns = [["relatives", "probands"], ["age", "age"]]
    display_df = pd.concat([display_df, N], axis=1)
    display_df = display_df[
        [("probands", "N"), ("probands", "age"),
         ("relatives", "N"), ("relatives", "age")]
    ]

    # Probands with relatives
    has_probands = df.groupby(["task", "sex", "id_nb"]).sum(numeric_only=True)["is_proband"]
    proband_idx = (
        has_probands.reset_index()
        .loc[has_probands.values == 1]
        .set_index(["task", "sex", "id_nb"])
        .index
    )
    relative_counts = (
        df.groupby(["task", "sex", "id_nb"]).count().loc[proband_idx, "site"] - 1
    )

    for threshold in [1, 2, 3]:
        col = ("Probands with relatives", f">={threshold}")
        display_df[col] = (
            (relative_counts >= threshold)
            .reset_index()
            .groupby(["task", "sex"])["site"]
            .sum()
        )

    percentages = (
        (display_df["Probands with relatives"].T / display_df[("probands", "N")].T).T * 100
    ).map(" ({:.1f}%)".format)
    display_df["Probands with relatives"] = (
        display_df["Probands with relatives"].astype(str) + percentages
    )

    mo.md("## Demographics breakdown")
    mo.ui.table(display_df)
    return (display_df,)


if __name__ == "__main__":
    app.run()
