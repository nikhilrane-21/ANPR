import torch
import cv2
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ops
from ultralytics.yolo.utils.plotting import Annotator, colors
import easyocr
import streamlit as st
import yaml
import os
import base64
st.set_page_config(page_title="ANPR", page_icon="🤖", layout="wide")
hide_streamlit_style = """
            <style>
            .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
            .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
            .viewerBadge_text__1JaDK {display: none;}
            MainMenu {visibility: hidden;}
            header { visibility: hidden; }
            footer {visibility: hidden;}
            #GithubIcon {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
st.title("Automatic Number Plate Recognition")
uploaded_file = st.file_uploader("Choose a video ", type=["mp4", "avi", "mkv", "mov"])
frame_placeholder = st.empty()

def ocr_image(img, coordinates):
    x, y, w, h = int(coordinates[0]), int(coordinates[1]), int(coordinates[2]), int(coordinates[3])
    img = img[y:h, x:w]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    result = reader.readtext(gray)
    text = ""
    for res in result:
        if len(result) == 1:
            text = res[1]
        if len(result) > 1 and len(res[1]) > 6 and res[2] > 0.2:
            text = res[1]
        return str(text)

reader = easyocr.Reader(['en'], gpu=True)


class DetectionPredictor(BasePredictor):
    count = 0

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()
        img /= 255
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds, self.args.conf, self.args.iou, agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)
        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        self.seen += 1
        im0 = im0.copy()
        self.data_path = p
        self.annotator = self.get_annotator(im0)
        det = preds[idx]
        for *xyxy, conf, cls in reversed(det):
            c = int(cls)
            text_ocr = ocr_image(im0, xyxy)
            label = text_ocr
            self.annotator.box_label(xyxy, label, color=colors(c, True))
            _, buffer = cv2.imencode('.jpg', im0)
            frame_base64 = base64.b64encode(buffer).decode()
            frame_placeholder.markdown(f'<img src="data:image/jpeg;base64,{frame_base64}"/>', unsafe_allow_html=True)



def load_config(config_path):
    with open(config_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

def predict(cfg, source_name):
    cfg['model'] = "bestn.pt"
    cfg['imgsz'] = [640, 640]
    cfg['source'] = source_name
    cfg['show'] = True
    predictor = DetectionPredictor(cfg)
    predictor()

if uploaded_file is not None:
    save_dir = "./uploads"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = uploaded_file.name
    file_path = os.path.join(save_dir, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    config_path = str(DEFAULT_CONFIG)
    cfg = load_config(config_path)
    predict(cfg, file_path)


