# Demo

<img src="example_data/IMG_0039_render.jpg?raw=true" width=400> <img src="example_data/IMG_0040_render.jpg?raw=true" width=400>
<img src="example_data/example-1-render.jpg?raw=true" width=400> <img src="example_data/example-1-render.jpg?raw=true" width=400 >
<img src="example_data/example-2-render.jpg?raw=true" width=400> <img src="example_data/example-2-render.jpg?raw=true" width=400 >


https://user-images.githubusercontent.com/17779555/229043335-a8e01496-0235-45c9-b5a4-e6227abd8c57.mp4

models used:
- bbox detector for finding clock face in the image
- classifier for clock orientation estimation
- keypoint detection for center and top
- semantic segmentation for finding clock hands
- KDE for splitting the binary segmentation mask into individual clock hands
### Watch crop with center and top keypoint
![Alt text](example_data/crop_and_center.jpg?raw=true "Watch crop with center and top")
### Detected mask of watch hands
![Alt text](example_data/hands_mask.jpg?raw=true "Detected mask of watch hands")
### KDE of pixel angles
![Alt text](example_data/debug_plots.jpg?raw=true "KDE of pixel angles")
### Fitted lines to segmented pixels
![Alt text](example_data/fitted_lines.jpg?raw=true "Fitted lines to segmented pixels")
### Final selected and rejected lines
![Alt text](example_data/selected_lines.jpg?raw=true "Selected and rejected lines")


# Metrics
| Path                           | val.1-min_acc   | val.10-min_acc   | val.60-min_acc   |
|--------------------------------|-----------------|------------------|------------------|
| metrics/end_2_end_summary.json | 0.224           | 0.345            | 0.414            |

| Path   |
|--------|

| Path   |
|--------|

| Path                      | eval.iou_score   | eval.loss   | step   | train.iou_score   | train.loss   |
|---------------------------|------------------|-------------|--------|-------------------|--------------|
| metrics/segmentation.json | 0.585            | 0.262       | 149    | 0.851             | 0.081        |

# Graph
```mermaid
flowchart TD
	node1["datasets/watch-faces.json.dvc"]
	node2["download-images"]
	node3["eval-detector"]
	node4["eval-end-2-end"]
	node5["eval-keypoint"]
	node6["eval-segmentation"]
	node7["export-detector"]
	node8["generate-detection-dataset"]
	node9["generate-watch-hands-dataset"]
	node10["train-detector"]
	node11["train-keypoint"]
	node12["train-segmentation"]
	node13["update-metrics"]
	node1-->node2
	node2-->node3
	node2-->node4
	node2-->node8
	node2-->node9
	node2-->node11
	node2-->node12
	node3-->node13
	node4-->node13
	node5-->node13
	node7-->node3
	node8-->node7
	node8-->node10
	node10-->node4
	node10-->node5
	node10-->node6
	node10-->node7
	node11-->node4
	node11-->node5
	node11-->node13
	node12-->node4
	node12-->node6
	node12-->node13
	node14["example_data/IMG_1200_720p.mov.dvc"]
	node15["render-demo"]
	node14-->node15
	node16["checkpoints/segmentation.dvc"]
	node17["checkpoints/detector.dvc"]
	node18["checkpoints/keypoint.dvc"]
```
# Installation
Install `watch_recognition` module, run pip in the main repository dir
```bash
pip install watch_recognition/
```
Tested on Python 3.7 and 3.8
## Running models
Checkout example notebook: `notebooks/demo-on-examples.ipynb`
## Models description
_TODO_
