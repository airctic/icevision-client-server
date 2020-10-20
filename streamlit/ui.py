import streamlit as st
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io

st.set_option("deprecation.showfileUploaderEncoding", False)

pennfundan_images = [
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/kids_crossing_street.jpg",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image0.png",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image1.png",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image2.png",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image3.png",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image4.png",
    "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image5.png",
    "https://www.adventisthealth.org/cms/thumbnails/00/1100x506/images/blog/kids_crossing_street.jpg",
    "https://i.cbc.ca/1.5510620.1585229177!/cumulusImage/httpImage/image.jpg_gen/derivatives/16x9_780/toronto-street-scene-covid-19.jpg",
]

# fastapi endpoint
# url = "http://fastapi:8000"
url = "http://127.0.0.1:8000"

endpoint = "/segmentation"


def sidebar_ui():
    # st.sidebar.image("images/airctic-logo-medium.png")
    st.sidebar.image(
        "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/icevision-deploy-small.png"
    )


# This sidebar UI lets the user select model thresholds.
def object_detector_ui():
    # st.sidebar.markdown("# Model Thresholds")
    detection_threshold = st.sidebar.slider("Detection threshold", 0.0, 1.0, 0.5, 0.01)
    mask_threshold = st.sidebar.slider("Mask threshold", 0.0, 1.0, 0.5, 0.01)
    return detection_threshold, mask_threshold


def process(uploaded_file, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", uploaded_file, "image/jpeg")})
    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )
    return r


def run_app():
    sidebar_ui()

    # Draw the threshold parameters for object detection model.
    detection_threshold, mask_threshold = object_detector_ui()

    bbox = st.sidebar.checkbox(label="Bounding Box", value=False)

    st.sidebar.image(
        "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/airctic-logo-medium.png"
    )

    st.markdown("### ** Insert an image**")
    uploaded_file = st.file_uploader("")  # image upload widget
    my_placeholder = st.empty()
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        my_placeholder.image(image, caption="", use_column_width=True)

    if st.button("Get Masks"):
        if uploaded_file is None:
            st.write("Insert an image!")  # handle case with no image
        else:
            segments = process(uploaded_file, url + endpoint)
            segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
            my_placeholder.image(segmented_image)


if __name__ == "__main__":
    run_app()
