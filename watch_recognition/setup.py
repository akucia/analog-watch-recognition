import setuptools

setuptools.setup(
    name="watch_recognition",
    version="0.0.1",
    author="Artur Kucia",
    author_email="author@example.com",
    description="Reading time from analog clocks",
    long_description_content_type="",
    url="",
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "tensorflow==2.6.*",
        "pandas==1.3.1",
        "numpy==1.19.5",
        "albumentations==1.0.3",
        "tqdm~=4.62.1",
        "matplotlib~=3.4.3",
        "scikit-image~=0.18.2",
        "Pillow==8.3.2",
        "scikit-learn~=0.24.1",
        "pycocotools",
        "click==8.0.1",
        "requests",
        "segmentation_models==1.0.1",
    ],
)
