update-dataset:
	python scripts/download-dataset.py 1 --export-file datasets/watch-faces.json

ingest-images:
	python scripts/add-images-to-label-studio-project.py --source-dir ./new-images/ --service-account-file ${SERVICE_ACCOUNT} --label-studio-project ${PROJECT_ID}