import marimo

__generated_with = "0.10.0"
app = marimo.App(width="full")


@app.cell
def parameters():
    # __Q1K_PARAMETERS__
    project_path = ""
    redcap_dir = ""
    sharepoint_path = ""
    mni_upload_date = ""
    hsj_upload_date = ""
    output_dir = ""
    return (project_path, redcap_dir, sharepoint_path,
            mni_upload_date, hsj_upload_date, output_dir)


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
        compare_with_sharepoint,
        generate_data_loss_excel,
        load_redcap,
        merge_tracking,
        scan_pipeline_stages,
    )
    return (mo, np, pd, go, warnings,
            PIPELINE_STAGES, VALID_TASKS,
            load_redcap, scan_pipeline_stages,
            merge_tracking, compare_with_sharepoint,
            generate_data_loss_excel)


@app.cell
def header(mo, project_path, sharepoint_path):
    mo.md(
        f"# Q1K Data Loss Comparison Report\n\n"
        f"**Project:** {project_path}\n\n"
        f"**SharePoint file:** {sharepoint_path}"
    )
    return ()


@app.cell
def load_data(load_redcap, redcap_dir):
    demographics_df = load_redcap(redcap_dir)
    print(f"Loaded {len(demographics_df)} subjects from REDCap")
    return (demographics_df,)


@app.cell
def primary_task_tracking(scan_pipeline_stages, merge_tracking,
                          demographics_df, project_path):
    """Track the primary task (RS) as the baseline."""
    pipeline_df = scan_pipeline_stages(project_path, "RS")
    tracking_df = merge_tracking(demographics_df, pipeline_df)
    print(f"RS tracking: {len(tracking_df)} subjects")
    return (tracking_df,)


@app.cell
def sharepoint_comparison(compare_with_sharepoint, tracking_df,
                          sharepoint_path, mni_upload_date,
                          hsj_upload_date, PIPELINE_STAGES):
    """Compare automated tracking with SharePoint data."""
    from pathlib import Path

    if not sharepoint_path or not Path(sharepoint_path).exists():
        print("No SharePoint file provided or file not found.")
        comparison_df = tracking_df.copy()
    else:
        comparison_df = compare_with_sharepoint(
            tracking_df,
            sharepoint_path,
            sheet_name="Resting state",
            mni_upload_date=mni_upload_date or None,
            hsj_upload_date=hsj_upload_date or None,
        )

    # Replace True/False with Yes/No for pipeline stages
    for stage in PIPELINE_STAGES:
        if stage in comparison_df.columns:
            comparison_df[stage] = comparison_df[stage].map(
                {True: "Yes", False: "No", "Yes": "Yes", "No": "No"}
            )

    return (comparison_df,)


@app.cell
def change_summary(mo, comparison_df):
    """Summarize changes detected."""
    if "changed" not in comparison_df.columns:
        mo.md("No comparison data available.")
        return ()

    changed = comparison_df[comparison_df["changed"] == "Yes"]
    unchanged = comparison_df[comparison_df["changed"] == "No"]

    mo.md(
        f"## Change Detection\n\n"
        f"- **Changed:** {len(changed)} subjects\n"
        f"- **Unchanged:** {len(unchanged)} subjects"
    )
    return ()


@app.cell
def status_breakdown(mo, pd, comparison_df):
    """Show breakdown of data loss reasons."""
    mo.md("## Status Breakdown")

    completed = comparison_df[
        comparison_df["Last_stage"] == "Completed"
    ]
    print(f"Completed: {len(completed)} "
          f"({len(completed)/len(comparison_df)*100:.1f}%)")

    if "Reason" in comparison_df.columns:
        reasons = comparison_df["Reason"].value_counts()
        print("\nData loss reasons:")
        for reason, count in reasons.items():
            if pd.notna(reason):
                print(f"  {reason}: {count}")
    return ()


@app.cell
def loss_chart(go, comparison_df):
    """Bar chart of data loss sources."""
    if "Reason" not in comparison_df.columns:
        return ()

    reasons = comparison_df["Reason"].dropna().value_counts()
    reasons = reasons[reasons.index != "Successful"]

    if len(reasons) == 0:
        return ()

    fig = go.Figure(data=[go.Bar(
        x=reasons.index.tolist(),
        y=reasons.values.tolist(),
    )])
    fig.update_layout(
        title="Data Loss Sources",
        xaxis_title="Reason",
        yaxis_title="Count",
    )
    fig
    return (fig,)


@app.cell
def other_tasks(mo, pd, scan_pipeline_stages, merge_tracking,
                demographics_df, project_path, VALID_TASKS,
                PIPELINE_STAGES):
    """Track all other tasks for cross-reference."""
    mo.md("## Other Tasks Summary")

    tasks = [t for t in VALID_TASKS if t not in ("RS", "RSRio")]
    task_summaries = {}
    for task in tasks:
        try:
            pdf = scan_pipeline_stages(project_path, task)
            mdf = merge_tracking(demographics_df, pdf)
            completed = len(
                mdf[mdf["Last_stage"] == "Completed"]
            )
            total = len(mdf[mdf["Last_stage"] != "None"])
            task_summaries[task] = {
                "Total with data": total,
                "Completed": completed,
            }
        except Exception as e:
            task_summaries[task] = {"Error": str(e)}

    summary = pd.DataFrame(task_summaries).T
    mo.ui.table(summary)
    return ()


@app.cell
def save_excel(generate_data_loss_excel, project_path, redcap_dir,
               sharepoint_path, output_dir, mni_upload_date,
               hsj_upload_date):
    """Save the data loss Excel report."""

    sp = sharepoint_path if sharepoint_path else None
    out = output_dir if output_dir else None

    out_path = generate_data_loss_excel(
        project_path=project_path,
        redcap_dir=redcap_dir,
        sharepoint_path=sp,
        output_dir=out,
        mni_upload_date=mni_upload_date or None,
        hsj_upload_date=hsj_upload_date or None,
    )
    print(f"Saved data loss Excel: {out_path}")
    return (out_path,)


if __name__ == "__main__":
    app.run()
