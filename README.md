# Demo

https://user-images.githubusercontent.com/17779555/136506927-d326381b-6d54-4c2a-91a8-aa0fee89ba36.mov


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
  <KeyPointLabels name="keypoint" toName="img" strokewidth="5">
    <Label value="Center" background="blue">
    </Label>
    <Label value="Top" background="red">
    </Label>
    <Label value="Hour" background="green">
    </Label>
    <Label value="Minute" background="orange">
    </Label>
  </KeyPointLabels>
  <RectangleLabels name="bbox" toName="img">
    <Label value="WatchFace"/>
  </RectangleLabels>
  <Image name="img" value="$image" zoom="true" zoomControl="true" width='100%' maxWidth='1500' >
  </Image>
</View>

```
References 
1. OpenImagesDataset https://opensource.google/projects/open-images-dataset
