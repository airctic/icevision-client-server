__all__ = [
    "load_model",
    "image_from_url",
    "predict",
    "get_masks",
]

from icevision.all import *

# from PIL import Image
import PIL, requests
import torch
from torchvision import transforms
import io


def load_model(class_map=class_map, url=None):
    if url is None:
        # print("Please provide a valid URL")
        return None
    else:
        model = mask_rcnn.model(num_classes=len(class_map))
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)
        return model


def image_from_url(url):
    res = requests.get(url, stream=True)
    img = PIL.Image.open(res.raw)
    return np.array(img)


def predict(
    model, 
    image, 
    detection_threshold: float = 0.5, 
    mask_threshold: float = 0.5, 
    display_label=True,
    display_bbox=True,
    display_mask=True,
):
    img = np.array(image)
    tfms_ = tfms.A.Adapter([tfms.A.Normalize()])
    # Whenever you have images in memory (numpy arrays) you can use `Dataset.from_images`
    infer_ds = Dataset.from_images([img], tfms_)

    batch, samples = mask_rcnn.build_infer_batch(infer_ds)
    preds = mask_rcnn.predict(
        model=model,
        batch=batch,
        detection_threshold=detection_threshold,
        mask_threshold=mask_threshold,
    )
    return samples[0]["img"], preds[0]


def get_masks(model, binary_image, 
    class_map=None, 
    detection_threshold: float = 0.5, 
    mask_threshold: float = 0.5, 
    display_label=True,
    display_bbox=True,
    display_mask=True,
    ):
    input_image = PIL.Image.open(io.BytesIO(binary_image)).convert("RGB")
    img, pred = predict(model=model, 
        image=input_image, 
        detection_threshold=detection_threshold, 
        mask_threshold=mask_threshold,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
    )

    img = draw_pred(
        img=img,
        pred=pred,
        class_map=class_map,
        denormalize_fn=denormalize_imagenet,
        display_label=display_label,
        display_bbox=display_bbox,
        display_mask=display_mask,
    )
    img = PIL.Image.fromarray(img)
    # print("Output Image: ", img.size, type(img))
    return img
