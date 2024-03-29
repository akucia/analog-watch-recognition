update-dataset:
	python scripts/download-dataset.py 1 --export-file datasets/watch-faces.json
	dvc add datasets/watch-faces.json

add-storage-to-label-studio:
	python scripts/add-google-storage.py --label-studio-project ${PROJECT_ID} --label-studio-host ${LABEL_STUDIO_URL} --label-studio-api-token ${LABEL_STUDIO_ACCESS_TOKEN} --service-account-file ${SERVICE_ACCOUNT}

update-image-cache: update-dataset
	dvc repro -s download-images

add-images:
	python scripts/add-images-to-label-studio-project.py --source-dir ./new-images-4 --label-studio-project ${PROJECT_ID} --label-studio-host ${LABEL_STUDIO_URL} --label-studio-api-token ${LABEL_STUDIO_ACCESS_TOKEN}  --n-images 50 --shuffle-images

update-metrics:
	python scripts/update-metrics-table-and-graph.py

dev-install:
	pip install -r dev-requirements.txt && pip install -e ./watch_recognition

select-unused-images:
	python scripts/select-unused-images.py --data-dir ./new-images --output-path ./unused-images --clean-save-dir

active-learning-select-images:
	python scripts/score-images-for-learning.py --data-dir ./unused-images --run-concurrently --num-samples 15 --output-path ./sampled-images --clean-save-dir

add-sampled-images:
	python scripts/add-images-to-label-studio-project.py --source-dir ./sampled-images --label-studio-project ${PROJECT_ID} --label-studio-host ${LABEL_STUDIO_URL} --label-studio-api-token ${LABEL_STUDIO_ACCESS_TOKEN}

generate-full-requirements:
	pip-compile -v dev-requirements.in.txt -o dev-requirements.txt

run-detection-evaluation:
	python watch_recognition/watch_recognition/eval/object_detection_eval.py --save-plots --save-per-image-metrics

render-demo-video:
	python scripts/render-demo-movie.py example_data/IMG_1200_720p.mov IMG_1200_720p-render.mp4
