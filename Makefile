update-dataset:
	python scripts/download-dataset.py 1 --export-file datasets/watch-faces.json

add-images:
	python scripts/add-images-to-label-studio-project.py --source-dir ./new-images/ --label-studio-project ${PROJECT_ID}