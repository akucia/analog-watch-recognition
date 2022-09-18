# Graph
```mermaid
flowchart TD
	node1[checkpoints/detector.dvc]
	node2[checkpoints/keypoint.dvc]
	node3[checkpoints/segmentation.dvc]
	node4[datasets/watch-faces.json.dvc]
	node5[download-images]
	node6[eval-detector]
	node7[eval-end-2-end]
	node8[eval-keypoint]
	node9[eval-segmentation]
	node10[train-detector]
	node11[train-keypoint]
	node12[train-segmentation]
	node13[update-metrics]
	node1-->node10
	node2-->node11
	node3-->node12
	node4-->node5
	node5-->node7
	node5-->node10
	node5-->node11
	node5-->node12
	node6-->node13
	node7-->node13
	node8-->node13
	node10-->node6
	node10-->node7
	node10-->node8
	node10-->node9
	node10-->node13
	node11-->node7
	node11-->node8
	node11-->node13
	node12-->node7
	node12-->node9
	node12-->node13
```
# Metrics
| Path                           | train.1-min_acc   | train.10-min_acc   | train.60-min_acc   | val.1-min_acc   | val.10-min_acc   | val.60-min_acc   |
|--------------------------------|-------------------|--------------------|--------------------|-----------------|------------------|------------------|
| metrics/end_2_end_summary.json | 0.097             | 0.179              | 0.306              | 0.026           | 0.026            | 0.052            |

| Path                             | AP @IoU=0.50   | AP @IoU=0.50:0.95   | AP @IoU=0.75   | AP @IoU=0.95   | AR @maxDets=1   | AR @maxDets=10   | AR @maxDets=100   | Num Images   | eval.Mean Average Precision   | eval.Recall   | eval.box_loss   | eval.classification_loss   | eval.loss   | eval.regularization_loss   | step   | train.box_loss   | train.classification_loss   | train.loss   | train.regularization_loss   |
|----------------------------------|----------------|---------------------|----------------|----------------|-----------------|------------------|-------------------|--------------|-------------------------------|---------------|-----------------|----------------------------|-------------|----------------------------|--------|------------------|-----------------------------|--------------|-----------------------------|
| metrics/detector.json            | -              | -                   | -              | -              | -               | -                | -                 | -            | 0.135                         | 0.292         | 3.216           | 6.612                      | 9.828       | 0.0                        | 59     | 0.229            | 0.012                       | 0.241        | 0.0                         |
| metrics/detector/coco_train.json | 0.864          | 0.779               | 0.864          | -1.0           | 0.869           | 0.927            | 0.927             | 218          | -                             | -             | -               | -                          | -           | -                          | -      | -                | -                           | -            | -                           |
| metrics/detector/coco_val.json   | 0.677          | 0.421               | 0.433          | -1.0           | 0.482           | 0.571            | 0.571             | 35           | -                             | -             | -               | -                          | -           | -                          | -      | -                | -                           | -            | -                           |

| Path                             | AP @IoU=0.50   | AP @IoU=0.50:0.95   | AP @IoU=0.75   | AR @IoU=0.50   | AR @IoU=0.50:0.95   | AR @IoU=0.75   | Num Images   | eval.iou_score   | eval.loss   | step   | train.iou_score   | train.loss   |
|----------------------------------|----------------|---------------------|----------------|----------------|---------------------|----------------|--------------|------------------|-------------|--------|-------------------|--------------|
| metrics/keypoint.json            | -              | -                   | -              | -              | -                   | -              | -            | 0.48             | 0.524       | 59     | 0.89              | 0.11         |
| metrics/keypoint/coco_train.json | 0.116          | 0.045               | 0.014          | 0.236          | 0.127               | 0.082          | 218          | -                | -           | -      | -                 | -            |
| metrics/keypoint/coco_val.json   | 0.241          | 0.146               | 0.136          | 0.357          | 0.257               | 0.25           | 35           | -                | -           | -      | -                 | -            |

| Path                      | eval.iou_score   | eval.loss   | step   | train.iou_score   | train.loss   |
|---------------------------|------------------|-------------|--------|-------------------|--------------|
| metrics/segmentation.json | 0.561            | 0.275       | 59     | 0.872             | 0.067        |

## End 2 end metrics definitions
Final metric for the entire system is 'x-min accuracy' which is the fraction of system predictions accurate within x minutes. Example:  
$$\text{1-min-acc} = 1 - {|{|time - {time}_{pred}| < 1min}| \over N_{samples}}$$
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
