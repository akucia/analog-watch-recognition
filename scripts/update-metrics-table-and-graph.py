import subprocess

if __name__ == "__main__":
    with open("README.md") as f:
        readme_lines = f.readlines()

    graph_start = readme_lines.index("# Graph\n")
    graph_end = readme_lines.index("# Metrics\n")
    readme_before_graph = readme_lines[:graph_start]
    graph_md = [
        subprocess.run(["dvc", "dag", "--md"], stdout=subprocess.PIPE).stdout.decode(
            "utf-8"
        )
    ]

    metrics_start = readme_lines.index("# Metrics\n")
    metrics_end = readme_lines.index("## End 2 end metrics definitions\n")
    metrics_tables_md = []
    commands = [
        [
            "dvc",
            "metrics",
            "show",
            "metrics/end_2_end_summary.json",
            "--precision",
            "3",
            "--md",
        ],
        [
            "dvc",
            "metrics",
            "show",
            "metrics/detector.json",
            "metrics/detector/coco_train.json",
            "metrics/detector/coco_val.json",
            "--precision",
            "3",
            "--md",
        ],
        [
            "dvc",
            "metrics",
            "show",
            "metrics/keypoint.json",
            "metrics/keypoint/coco_train.json",
            "metrics/keypoint/coco_val.json",
            "--precision",
            "3",
            "--md",
        ],
        [
            "dvc",
            "metrics",
            "show",
            "metrics/segmentation.json",
            "--precision",
            "3",
            "--md",
        ],
    ]
    for cmd in commands:
        print(" ".join(cmd))
        metrics_tables_md.append(
            subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")
        )

    readme_after_metrics = readme_lines[metrics_end:]

    graph_header = [readme_lines[graph_start]]
    metrics_header = [readme_lines[metrics_start]]
    new_readme = (
        readme_before_graph
        + graph_header
        + graph_md
        + metrics_header
        + metrics_tables_md
        + readme_after_metrics
    )

    with open("README.md", "w") as f:
        f.writelines(new_readme)
