from enum import Enum, auto
from pathlib import Path

import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

from watch_recognition.eval.end_to_end_eval import _evaluate_on_single_image
from watch_recognition.predictors import (
    HandPredictorGRPC,
    KPHeatmapPredictorV2GRPC,
    RetinaNetDetectorGRPC,
    TimePredictor,
)

model_host = "localhost:8500"
time_predictor = TimePredictor(
    detector=RetinaNetDetectorGRPC(
        host=model_host,
        model_name="detector",
        class_to_label_name={0: "WatchFace"},
    ),
    kp_predictor=KPHeatmapPredictorV2GRPC(
        host=model_host,
        model_name="keypoint",
        class_to_label_name={
            0: "Top",
            1: "Center",
            2: "Crown",
        },
        confidence_threshold=0.5,
    ),
    hand_predictor=HandPredictorGRPC(
        host=model_host,
        model_name="segmentation",
    ),
)


class Menu(Enum):
    demo = auto()
    debug = auto()


def demo():
    file = st.file_uploader("Input image")
    if file:
        demo_on_file(file)


def demo_on_file(file):
    image = Image.open(file)
    image = ImageOps.exif_transpose(image)
    image.convert("RGB")
    bboxes = time_predictor.predict(image)
    if not bboxes:
        st.warning("No results")
    df = pd.DataFrame(bboxes)
    df
    fig, ax = plt.subplots()
    plt.tight_layout()
    plt.axis("off")
    ax.imshow(image)
    for bbox in bboxes:
        bbox.plot(ax=ax)
    st.pyplot(fig)


def debug():
    input_file = st.file_uploader("Input image")
    example_id = st.text_input("Enter dataset example id")
    use_file = st.checkbox("file input")
    if input_file and use_file:
        file = input_file
    elif example_id:
        # TODO load example from dataset
        example_id = int(example_id)
        source = Path("./datasets/watch-faces-local.json")
        eval_result = _evaluate_on_single_image(
            example_id,
            time_predictor,
            source,
        )
        dataset_example = pd.DataFrame(eval_result)
        dataset_example
        file = Path(eval_result[0]["image_path"])
    else:
        st.markdown("Please specify file or example id")
        return

    st.markdown("## Detections")
    if file:
        demo_on_file(file)
    with Image.open(file) as img:
        fig, ax = plt.subplots()
        plt.tight_layout()
        plt.axis("off")
        time_predictor.detector.predict_and_plot(img, ax=ax)
        st.pyplot(fig)
        # todo show crops and stages of every prediction
        bboxes = time_predictor.detector.predict(img)
        for i, bbox in enumerate(bboxes):
            st.markdown(f"### result {i}")
            with img.crop(box=bbox.as_coordinates_tuple) as crop:
                fig, axarr = plt.subplots(1, 2)
                fig.tight_layout()
                for ax in axarr:
                    ax.axis("off")
                time_predictor.kp_predictor.predict_and_plot(crop, ax=axarr[0])
                axarr[0].legend()
                time_predictor.hand_predictor.predict_mask_and_draw(crop, ax=axarr[1])
                st.pyplot(fig)


def main():
    with st.sidebar:
        selection = st.radio("Choose a mode ", list(Menu.__members__))
    if selection == Menu.demo.name:
        st.markdown("# Demo")
        demo()
    elif selection == Menu.debug.name:
        st.markdown("# Debug")
        debug()
    else:
        st.warning(f"unknown mode {selection}")


if __name__ == "__main__":
    main()
