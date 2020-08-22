from mantisshrimp.all import *
from fastapi import FastAPI, File
from starlette.responses import Response
import io

# from fastapi.masks import *

MASK_PENNFUNDAN_WEIGHTS_URL = "https://mantisshrimp-models.s3.us-east-2.amazonaws.com/pennfundan_maskrcnn_resnet50fpn.zip"
# img_url = "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image2.png"

# class_map = datasets.pennfundan.class_map()
# model = load_model(class_map=class_map, url=MASK_PENNFUNDAN_WEIGHTS_URL)

app = FastAPI(
    title="IceVision Object Dectection",
    description="""Perform Mask RCNN Dectection using IceVision Framework.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)


@app.post("/segmentation/{img_url}")
def get_predicted_image(img_url: str):
    """Get masks from image"""
    return f"Hello my {img_url}"

    # segmented_image = get_masks(model, img_url, class_map=class_map)
    # bytes_io = io.BytesIO()
    # segmented_image.save(bytes_io, format="PNG")
    # return Response(bytes_io.getvalue(), media_type="image/png")
