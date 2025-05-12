import cv2
import torch
from model.base import fast_glcm
import numpy as np
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib import pyplot as plt


device = torch.device('cpu')
print(f"Using device: {device}")
class CamProcessor:
    def __init__(self, model_path, input_size=(512, 512), cam_per=0.4, glcm_per=0.6, Threshold=0.4):
        try:
            self.model = torch.load(model_path, map_location='cpu',weights_only=False).eval()
        except Exception as e:
            raise RuntimeError(f"Model loading failure: {str(e)}")
        self.image_size = input_size
        self.cam_per = cam_per
        self.glcm_per = glcm_per
        self.Threshold = Threshold

    def process_image(self, image_path):
        img, ori_size = self._load_and_preprocess(image_path)
        # plt.imsave("./temp/load.png", img)#正常
        cam = self._generate_cam(img)
        glcm = self._generate_glcm(img)
        # plt.imsave("./temp/cam.png", cam)
        diffused_cam, binary_mask = self._fuse_features(cam, glcm)
        # plt.imsave("./temp/fuse.png", diffused_cam)
        overlay_img = self._create_overlay(img, diffused_cam)
        transparent_img = self._create_transparent(img, diffused_cam)
        boundary_img = self._create_boundary(img, binary_mask)
        entropy = self.imageEntropy(overlay_img)
        mr = self.MineralizationRatio(overlay_img)

        return cv2.resize(binary_mask, ori_size), cv2.resize(overlay_img, ori_size), \
            cv2.resize(transparent_img, ori_size), cv2.resize(boundary_img, ori_size), entropy, mr

    def _load_and_preprocess(self, image_path):
        img = cv2.imread(image_path)[:, :, ::-1]
        h, w = img.shape[:2]
        return cv2.resize(img, self.image_size), (w, h)

    def _generate_cam(self, img):
        tensor = torch.tensor(np.expand_dims(img, 0)).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            cam = self.model(tensor).squeeze().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min())

    def _generate_glcm(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        glcm = fast_glcm.fast_glcm_mean(gray)
        return glcm / glcm.max()

    def _fuse_features(self, cam, glcm):
        diffused = self.glcm_per * glcm + self.cam_per * cam
        overlay_img = np.where(diffused < self.Threshold, 0, diffused).astype(np.float32)
        binary_mask = np.where(diffused > self.Threshold, 1, 0).astype(np.uint8)
        return overlay_img, binary_mask

    def _create_overlay(self, img, mask):
        norm_img = img / 255.0
        return show_cam_on_image(norm_img, mask, use_rgb=True)

    def _create_transparent(self, img, mask):
        rgba = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        rgba[:, :, 3] = (mask * 255).astype(np.uint8)
        return rgba

    def _create_boundary(self, original_img, mask, glow_color=(255, 150, 0),
                             glow_size=15, glow_intensity=0.7, kernel_size=3):
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        mask_255 = mask * 255

        if original_bgr.shape[:2] != mask_255.shape:
            mask_255 = cv2.resize(mask_255, (original_bgr.shape[1], original_bgr.shape[0]))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        boundary = cv2.morphologyEx(mask_255, cv2.MORPH_GRADIENT, kernel)

        glow_layer = np.zeros_like(original_bgr)
        glow_layer[boundary == 255] = glow_color

        glow_size = glow_size + 1 if glow_size % 2 == 0 else glow_size
        blurred_glow = cv2.GaussianBlur(glow_layer, (glow_size, glow_size), 0)

        blended = cv2.addWeighted(original_bgr, 1.0, blurred_glow, glow_intensity, 0)
        blended[boundary == 255] = glow_color
        return blended

    def imageEntropy(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

        entropy = 0
        for i in range(256):
            p = hist[i] / gray.size
            if p != 0:
                entropy += -p * np.log2(p)
        return np.float32(entropy[0])

    def MineralizationRatio(self, gray_heatMap, thres=0.5):
        shape = gray_heatMap.shape
        total = shape[0] * shape[1]
        ret, binary = cv2.threshold(gray_heatMap, thres * 255, 255, cv2.THRESH_BINARY)
        pixel_num = np.sum(binary == 0)
        return np.float32(pixel_num) / total


# 使用示例
if __name__ == "__main__":
    processor = CamProcessor("/mnt/windows_F/wyj_project/VRP-SAM-main/model/base/camnet.pth")

    mask, overlay, transparent, boundary, _, _ = processor.process_image("/mnt/windows_F/wyj_project/VRP-SAM-main/dataset/ROCK/rock_orgin/test (copy)/XPL+/16A0631_a1.png")

    cv2.imwrite("./mask/mask.png", mask * 255)
    cv2.imwrite("./mask/overlay.png", overlay[:, :, ::-1])
    cv2.imwrite("./mask/transparent.png", transparent)
    cv2.imwrite("./mask/boundary.png", boundary)

