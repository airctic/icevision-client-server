from icevision.all import *
from fastapi import FastAPI, File
from starlette.responses import Response
import io

from masks import *

MASK_PENNFUNDAN_WEIGHTS_URL = "https://github.com/airctic/model_zoo/releases/download/pennfudan_maskrcnn_resnet50fpn/pennfudan_maskrcnn_resnet50fpn.zip"
# img_url = "https://raw.githubusercontent.com/ai-fast-track/ice-streamlit/master/images/image2.png"

class_map = icedata.pennfudan.class_map()
model = load_model(class_map=class_map, url=MASK_PENNFUNDAN_WEIGHTS_URL)
print("class_map: ", class_map)

app = FastAPI(
    title="IceVision Object Dectection",
    description="""Perform Mask RCNN Dectection using IceVision Framework.
                           Visit this URL at port 8501 for the streamlit interface.""",
    version="0.1.0",
)


@app.post("/segmentation")
def get_predicted_image(file: bytes = File(...)):
    """Get masks from image"""
    segmented_image = get_masks(model, file, class_map=class_map)
    bytes_io = io.BytesIO()
    segmented_image.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")
