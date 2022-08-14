import subprocess

if __name__ == "__main__":
    with open("README.md") as f:
        readme_lines = f.readlines()

    metrics_start = readme_lines.index("# Metrics\n")
    metrics_end = readme_lines.index("# Demo - version 2\n")
    cmd = ["dvc", "metrics", "show", "--md"]
    output_metrics = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")
    cmd = ["dvc", "dag", "--md"]
    output_graph = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode("utf-8")

    readme_before_metrics = readme_lines[:metrics_start]
    new_metrics = [readme_lines[metrics_start]] + [output_metrics, output_graph]
    readme_after_metrics = readme_lines[metrics_end:]

    new_readme = readme_before_metrics + new_metrics + readme_after_metrics

    with open("README.md", "w") as f:
        f.writelines(new_readme)
