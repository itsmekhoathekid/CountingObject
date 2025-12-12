import sys
import os
HOME= os.getcwd()
path = os.path.abspath(os.path.join(HOME, 'CountingObject/models/GroundingDINO'))
sys.path.append(path)


from GroundingDINO.groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate
)
import supervision as sv

DIR_WEIGHTS = os.path.join(HOME, "CountingObject/models/pretrained_models")
CONFIG_PATH = os.path.join(HOME, "CountingObject/models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
WEIGHTS_PATH = os.path.join(DIR_WEIGHTS, "groundingdino_swint_ogc.pth")

class GetExampler:
    def __init__(self, device='cuda'):
        self.download_model()
        self.model = load_model(CONFIG_PATH, WEIGHTS_PATH, device)
        self.model = self.model.to(device)

    def download_model(self):
        os.makedirs(DIR_WEIGHTS, exist_ok=True)
        if os.path.exists(WEIGHTS_PATH):
            print(f"Model weights already exist at {WEIGHTS_PATH}. Skipping download.")
        else:
            import urllib.request
            print("Downloading model weights...")
            url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
            urllib.request.urlretrieve(url, WEIGHTS_PATH)
            print("Saved to:", WEIGHTS_PATH)



    def get_exampler(self, image_path, caption, box_threshold=0.35, text_threshold=0.25, device='cuda'):
        imag_source, image_transformed = load_image(image_path)
        boxes, logits, phrases = predict(
            self.model,
            image_transformed,
            caption,
            box_threshold,
            text_threshold,
            device,
            remove_combined=False
        )
        return boxes, logits, phrases, imag_source
import cv2

def test():
    image_path = "/home/anhkhoa/anhkhoa/CountingObject/Dataset/images_384_VarV2/285.jpg"

    get_exampler = GetExampler()
    boxes, logits, phrases, img_source = get_exampler.get_exampler(
        image_path=image_path,
        caption="strawberry"
    )

    annotated = annotate(image_source=img_source, boxes=boxes, logits=logits, phrases=phrases)

    out_path = "/home/anhkhoa/anhkhoa/CountingObject/examples/debug_groundingdino.jpg"
    cv2.imwrite(out_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    print("Saved:", out_path)

if __name__ == "__main__":
    test()

    