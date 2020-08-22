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
url = "http://fastapi:8000"
endpoint = "/segmentation"


def process(uploaded_file, server_url: str):
    m = MultipartEncoder(fields={"file": ("filename", uploaded_file, "image/jpeg")})
    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )
    return r


st.title("IceVision Web App")

st.write(
    """Obtain semantic segmentation maps of the image in input via DeepLabV3 implemented in PyTorch.
         This streamlit example uses a FastAPI service as backend.
         Visit this URL at `:8000/docs` for FastAPI documentation."""
)  # description and instructions

# st.markdown("### ** Paste Your Image URL**")
# my_placeholder = st.empty()

# index = 0
# image_path = pennfundan_images[index]
# image_url_key = f"image_url_key-{index}"
# image_url = my_placeholder.text_input(label="", value=image_path, key=image_url_key)

uploaded_file = st.file_uploader("insert image")  # image upload widget
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)


if st.button("Get segmentation map"):
    if uploaded_file is None:
        st.write("Insert an image!")  # handle case with no image
    else:
        segments = process(uploaded_file, url + endpoint)
        segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
        st.image([image, segmented_image], width=300)  # output dyptich


# def process_img_url(server_url: str, img_url: str):

#     full_url = f"{server_url}/{img_url}"
#     r = requests.post(full_url, headers={"Content-Type": "text"}, timeout=8000)

#     return r


# if st.button("Get Masks"):

#     if image_url is None:
#         st.write("Insert an image URL!")
#     else:
#         segments = process_img_url(url + endpoint, image_url)
#         segmented_image = Image.open(io.BytesIO(segments.content)).convert("RGB")
#         st.image(segmented_image)
