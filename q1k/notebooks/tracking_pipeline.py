import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def parameters():
    # __Q1K_PARAMETERS__
    project_path = ""
    redcap_dir = ""
    output_dir = ""
    return project_path, redcap_dir, output_dir


@app.cell
def imports():
    import warnings

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    warnings.filterwarnings("ignore")

    from q1k.config import PIPELINE_STAGES, VALID_TASKS
    from q1k.tracking.tools import (
        generate_tracking_csv,
        load_redcap,
        merge_tracking,
        scan_pipeline_stages,
    )
    return (mo, np, pd, go, warnings,
            PIPELINE_STAGES, VALID_TASKS,
            load_redcap, scan_pipeline_stages,
            merge_tracking, generate_tracking_csv)


@app.cell
def header(mo, project_path):
    mo.md(
        f"# Q1K Pipeline Tracking Dashboard\n\n"
        f"**Project:** {project_path}"
    )
    return ()


@app.cell
def load_data(load_redcap, redcap_dir):
    demographics_df = load_redcap(redcap_dir)
    print(f"Loaded {len(demographics_df)} subjects from REDCap")
    return (demographics_df,)


@app.cell
def task_completion_summary(mo, pd, demographics_df, VALID_TASKS):
    """Show REDCap task completion counts."""
    tasks = [t for t in VALID_TASKS if t != "RSRio"]
    counts = {}
    for task in tasks:
        if task in demographics_df.columns:
            counts[task] = int(demographics_df[task].sum())
        else:
            counts[task] = 0

    summary_df = pd.DataFrame({
        "Task": list(counts.keys()),
        "Completed (REDCap)": list(counts.values()),
    })

    mo.md("## REDCap Task Completion Counts")
    return (summary_df,)


@app.cell
def show_completion_table(mo, summary_df):
    mo.ui.table(summary_df)
    return ()


@app.cell
def select_task(mo, VALID_TASKS):
    task_selector = mo.ui.dropdown(
        options=[t for t in VALID_TASKS if t != "RSRio"],
        value="RS",
        label="Select task for pipeline tracking:",
    )
    task_selector
    return (task_selector,)


@app.cell
def scan_task(scan_pipeline_stages, project_path, task_selector):
    selected_task = task_selector.value
    pipeline_df = scan_pipeline_stages(project_path, selected_task)
    print(
        f"Found {len(pipeline_df)} subjects with data "
        f"for task {selected_task}"
    )
    return pipeline_df, selected_task


@app.cell
def merge_data(merge_tracking, demographics_df, pipeline_df,
               PIPELINE_STAGES):
    tracking_df = merge_tracking(demographics_df, pipeline_df)

    # Stage counts
    stage_counts = {}
    for stage in PIPELINE_STAGES:
        if stage in tracking_df.columns:
            stage_counts[stage] = int(tracking_df[stage].sum())

    print("Subjects per pipeline stage:")
    for stage, count in stage_counts.items():
        print(f"  {stage}: {count}")

    return tracking_df, stage_counts


@app.cell
def sankey_diagram(go, stage_counts, PIPELINE_STAGES, selected_task):
    """Create a Sankey diagram showing data flow through the pipeline."""
    stages = [s for s in PIPELINE_STAGES if s in stage_counts]
    if len(stages) < 2:
        print("Not enough data for Sankey diagram")
        return ()

    labels = stages + [f"Lost after {s}" for s in stages[:-1]]
    source = []
    target = []
    value = []

    for i in range(len(stages) - 1):
        current = stage_counts[stages[i]]
        next_count = stage_counts[stages[i + 1]]
        lost = current - next_count

        # Flow to next stage
        source.append(i)
        target.append(i + 1)
        value.append(max(next_count, 0))

        # Flow to loss node
        if lost > 0:
            loss_idx = len(stages) + i
            source.append(i)
            target.append(loss_idx)
            value.append(lost)

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=labels,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
        ),
    )])
    fig.update_layout(
        title_text=f"Pipeline Data Flow — {selected_task}",
        font_size=12,
    )
    fig
    return (fig,)


@app.cell
def discrepancy_check(mo, tracking_df):
    """Flag subjects with non-contiguous pipeline progression."""
    disc = tracking_df[tracking_df["discrepancy"]]
    if len(disc) > 0:
        mo.md(
            f"**Warning:** {len(disc)} subjects have "
            f"non-contiguous pipeline progression (skipped steps)."
        )
        mo.ui.table(
            disc[["q1k_ID", "bids_id", "Last_stage", "site"]]
        )
    else:
        mo.md("No discrepancies detected in pipeline progression.")
    return ()


@app.cell
def demographics_summary(mo, tracking_df):
    """Show demographics breakdown."""
    mo.md("## Demographics Summary")
    if "group" in tracking_df.columns:
        group_counts = tracking_df["group"].value_counts()
        print("By family role:")
        for group, count in group_counts.items():
            print(f"  {group}: {count}")
    if "site" in tracking_df.columns:
        site_counts = tracking_df["site"].value_counts()
        print("\nBy site:")
        for site, count in site_counts.items():
            print(f"  {site}: {count}")
    return ()


@app.cell
def save_output(generate_tracking_csv, project_path, selected_task,
                redcap_dir, output_dir):
    """Save tracking CSV for the selected task."""
    out_dir = output_dir if output_dir else None
    out_path = generate_tracking_csv(
        project_path, selected_task, redcap_dir, out_dir,
    )
    print(f"Saved tracking CSV: {out_path}")
    return (out_path,)


if __name__ == "__main__":
    app.run()
