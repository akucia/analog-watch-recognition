import subprocess  # nosec

if __name__ == "__main__":
    with open("README.md") as f:  # nosec
        readme_lines = f.readlines()

    graph_start = readme_lines.index("# Graph\n")
    graph_end = readme_lines.index("# Installation\n")
    metrics_start = readme_lines.index("# Metrics\n")
    metrics_end = graph_start
    readme_before_metrics = readme_lines[:metrics_start]

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
    graph_md = [  # nosec
        subprocess.run(["dvc", "dag", "--md"], stdout=subprocess.PIPE).stdout.decode(
            "utf-8"
        )
    ]
    for cmd in commands:
        print(" ".join(cmd))
        metrics_tables_md.append(  # nosec
            subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")
        )

    readme_after_graph = readme_lines[graph_end:]

    graph_header = [readme_lines[graph_start]]
    metrics_header = [readme_lines[metrics_start]]
    new_readme = (
        readme_before_metrics
        + metrics_header
        + metrics_tables_md
        + graph_header
        + graph_md
        + readme_after_graph
    )

    with open("README.md", "w") as f:
        f.writelines(new_readme)
