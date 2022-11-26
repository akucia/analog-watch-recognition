update-dataset:
	python scripts/download-dataset.py 1 --export-file datasets/watch-faces.json
	dvc add datasets/watch-faces.json

add-storage-to-label-studio:
	python scripts/add-google-storage.py --label-studio-project ${PROJECT_ID} --label-studio-host ${LABEL_STUDIO_URL} --label-studio-api-token ${LABEL_STUDIO_ACCESS_TOKEN} --service-account-file ${SERVICE_ACCOUNT}

update-image-cache: update-dataset
	dvc repro -s download-images

add-images:
	python scripts/add-images-to-label-studio-project.py --source-dir ./new-images --label-studio-project ${PROJECT_ID} --label-studio-host ${LABEL_STUDIO_URL} --label-studio-api-token ${LABEL_STUDIO_ACCESS_TOKEN}  --n-images 50 --shuffle-images

update-metrics:
	python scripts/update-metrics-table-and-graph.py

generate-detector-checkpoint:
	python watch_recognition/watch_recognition/train/object_detection_task.py --epochs 50 --batch-size 4 --confidence-threshold 0.5 --seed 42

generate-keypoint-checkpoint:
	python watch_recognition/watch_recognition/train/heatmap_regression_task.py --epochs 50 --batch-size 4 --confidence-threshold 0.5 --seed 42

dev-install:
	pip install -r dev-requirements.txt && pip install -e ./watch_recognition