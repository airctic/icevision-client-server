__all__ = [
    "load_model",
    "image_from_url",
    "predict",
    "show_prediction",
    "get_masks",
]

from mantisshrimp.all import *
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
    model, image_url, detection_threshold: float = 0.5, mask_threshold: float = 0.5
):
    img = image_from_url(image_url)

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


def show_prediction(img, pred, bbox=False, class_map=None):
    """Returns a PIL image"""
    show_pred(
        img=img,
        pred=pred,
        class_map=class_map,
        denormalize_fn=denormalize_imagenet,
        show=True,
        bbox=bbox,
    )

    # Grab image from the current matplotlib figure
    fig = plt.gcf()
    fig.canvas.draw()
    fig_arr = np.array(fig.canvas.renderer.buffer_rgba())
    img = PIL.Image.fromarray(fig_arr)

    return img

def get_masks(model, img_url, class_map=None):
    if img_url:
        img, pred = predict(
            model=model,
            image_url=img_url
        )
        return show_prediction(img=img, pred=pred, class_map=class_map)
