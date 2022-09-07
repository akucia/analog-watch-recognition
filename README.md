# Metrics
| Path                             | AP @IoU=0.50   | AP @IoU=0.50:0.95   | AP @IoU=0.75   | AP @IoU=0.95   | AR @IoU=0.50   | AR @IoU=0.50:0.95   | AR @IoU=0.75   | AR @maxDets=1   | AR @maxDets=10   | AR @maxDets=100   | Num Images   | eval.iou_score   | eval.loss   | step   | train.iou_score   | train.loss   |
|----------------------------------|----------------|---------------------|----------------|----------------|----------------|---------------------|----------------|-----------------|------------------|-------------------|--------------|------------------|-------------|--------|-------------------|--------------|
| metrics/detector.json            | -              | -                   | -              | -              | -              | -                   | -              | -               | -                | -                 | -            | -                | 0.42029     | 59     | -                 | 0.04062      |
| metrics/detector/coco_train.json | 0.45856        | 0.32698             | 0.40273        | -1.0           | -              | -                   | -              | 0.42            | 0.455            | 0.455             | 127          | -                | -           | -      | -                 | -            |
| metrics/detector/coco_val.json   | 1.0            | 0.7297              | 0.83168        | -1.0           | -              | -                   | -              | 0.75            | 0.75             | 0.75              | 6            | -                | -           | -      | -                 | -            |
| metrics/keypoint.json            | -              | -                   | -              | -              | -              | -                   | -              | -               | -                | -                 | -            | 0.62028          | 0.37972     | 59     | 0.89873           | 0.10617      |
| metrics/keypoint/coco_train.json | 0.52226        | 0.36543             | 0.31471        | -              | 0.65           | 0.50333             | 0.45833        | -               | -                | -                 | 127          | -                | -           | -      | -                 | -            |
| metrics/keypoint/coco_val.json   | 1.0            | 0.75399             | 0.63168        | -              | 1.0            | 0.78333             | 0.66667        | -               | -                | -                 | 6            | -                | -           | -      | -                 | -            |
| metrics/segmentation.json        | -              | -                   | -              | -              | -              | -                   | -              | -               | -                | -                 | -            | 0.44843          | 0.38081     | 59     | 0.74792           | 0.11398      |

```mermaid
flowchart TD
	node1[checkpoints/detector.dvc]
	node2[checkpoints/keypoint.dvc]
	node3[checkpoints/segmentation.dvc]
	node4[datasets/watch-faces.json.dvc]
	node5[download-images]
	node6[eval-detector]
	node7[eval-keypoint]
	node8[train-detector]
	node9[train-keypoint]
	node10[train-segmentation]
	node11[update-metrics]
	node1-->node8
	node2-->node9
	node3-->node10
	node4-->node5
	node5-->node8
	node5-->node9
	node5-->node10
	node6-->node11
	node7-->node11
	node8-->node6
	node8-->node7
	node8-->node11
	node9-->node7
	node9-->node11
	node10-->node11
```
# Demo - version 2

<img src="example_data/IMG_0039_render.jpg?raw=true" width=400> <img src="example_data/IMG_0040_render.jpg?raw=true" width=400>

<img src="example_data/Zrzut%20ekranu%202021-08-25%20o%2022.24.14_render.jpg?raw=true" width=400> <img src="example_data/Zrzut%20ekranu%202021-08-25%20o%2022.24.24_render.jpg?raw=true" width=400 >


https://user-images.githubusercontent.com/17779555/151705227-a892424b-279c-4a43-9076-494a55717d0c.mov

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

## Installation
Install `watch_recognition` module, run pip in the main repository dir
```bash
pip install watch_recognition/
```
Tested on Python 3.7 and 3.8
## Running models
Checkout example notebook: `notebooks/demo-on-examples.ipynb`
## Models description
_TODO_

# Demo - version 1

https://user-images.githubusercontent.com/17779555/136506927-d326381b-6d54-4c2a-91a8-aa0fee89ba36.mov

models used:
- bbox detector for finding clock face in the image
- classifier for clock orientation
- keypoint detection for center, top and end of clock hands

# Downloading images from OpenImage Dataset

```bash
wget https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
```

```bash
python scripts/downloader.py ./download_data/train_ids_small.txt --download_folder=./download_data/train/
```

```bash
python scripts/downloader.py ./download_data/test_ids_small.txt --download_folder=./download_data/test/
```

```bash
python scripts/downloader.py ./download_data/validation_ids_small.txt --download_folder=./download_data/validation/
```
# Convert tagged data into keypoint dataset

see notebook `./notebooks/generate_kp_dataset.ipynb`

# Train keypoint detection model
see notebook `./notebooks/cell-coder.ipynb.ipynb`

# Label Studio setup
https://labelstud.io/

```xml
<View>
    <Image name="image" value="$image" zoom="true" zoomControl="true"/>
      <KeyPointLabels name="kp" toName="image">
        <Label value="Center" background="#FFA39E"/>
        <Label value="Top" background="#D4380D"/>
        <Label value="Crown" background="#FFC069"/>
    </KeyPointLabels>
    <PolygonLabels name="polygon" toName="image" strokeWidth="3" pointSize="small" opacity="0.9">
        <Label value="Hands" background="#45fc03"/>
    </PolygonLabels>
    <RectangleLabels name="bbox" toName="image">
        <Label value="WatchFace" background="#FFA39E"/>
    </RectangleLabels>
      <TextArea name="transcription" toName="image" editable="true" perRegion="true" required="false" maxSubmissions="1" rows="5" placeholder="Recognized Time" displayMode="region-list"/>    
</View>
```
References 
1. OpenImagesDataset https://opensource.google/projects/open-images-dataset
