from SDC_ui import Ui_MainWindow
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QFileDialog, QApplication, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
import os
import torch
import cv2
import numpy as np
import random
import einops
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.openpose import OpenposeDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import config
import re
from torchvision import transforms
from annotator.canny import CannyDetector
from annotator.openpose import OpenposeDetector

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    
    def setup_control(self):
        self.ui.condiction_pth.currentIndexChanged.connect(self.get_condiction)
        self.ui.condiction_confirm.clicked.connect(self.set_model)
        self.ui.condiction_clear.clicked.connect(self.clear_model)
        self.ui.open_camera.clicked.connect(self.display_video)
        self.ui.snapshot.clicked.connect(self.snapshot_flag)
        self.ui.main_prompt_confirm.clicked.connect(self.get_prompt)
        self.ui.main_prompt_clear.clicked.connect(self.clear_prompt)
        self.ui.augmented_prompt_confirm.clicked.connect(self.get_a_prompt)
        self.ui.augmented_prompt_clear.clicked.connect(self.clear_a_prompt)
        self.ui.negative_prompt_confirm.clicked.connect(self.get_n_prompt)
        self.ui.negative_prompt_clear.clicked.connect(self.clear_n_prompt)
        self.ui.generate.clicked.connect(self.get_result)
        
        #
        self.ui.condiction_image.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.condiction_image.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.display_stream.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.display_stream.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.display_snapshot.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.display_snapshot.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.display_result.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.display_result.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        self.ui.ddim_step_slider.setRange(0, 50)
        self.ui.ddim_step_slider.setValue(25)
        self.ui.ddim_step_slider.valueChanged.connect(self.update_ddim_steps)
        
        self.ui.condiction_strength_slider.setRange(0, 200)
        self.ui.condiction_strength_slider.setValue(100)
        self.ui.condiction_strength_slider.valueChanged.connect(self.update_strength)
        
        self.ui.prompt_strength_slider.setRange(0, 30)
        self.ui.prompt_strength_slider.setValue(15)
        self.ui.prompt_strength_slider.valueChanged.connect(self.update_scale)
        
        self.ui.eta_slider.setRange(0, 100)
        self.ui.eta_slider.setValue(50)
        self.ui.eta_slider.valueChanged.connect(self.update_eta)
        
        self.ui.main_prompt_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.augmented_prompt_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.ui.negative_prompt_text.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        #
        self.ui.condiction_pth.addItems(['-', 'Canny Edge', 'Pose'])
        self.dict_pths = {
            'Canny Edge' : 'control_sd15_canny.pth',
            'Pose' : 'control_sd15_openpose.pth'
        }
        
        self.dict_condiction_img = {
            'Canny Edge' : 'canny_edge.jpg',
            'Pose' : 'pose.jpg'
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.WIDTH = 640
        self.HEIGHT = 480
        self.snapshot_status = False
        
        #
        self.src_img = None
        self.prompt = None
        self.a_prompt = None
        self.n_prompt = None
        self.num_samples = 1
        self.image_resolution = 512
        self.detect_resolution = 512
        self.ddim_steps = 25
        self.guess_mode = False
        self.strength = 1.0
        self.scale = 9
        self.seed = -1
        self.eta = 0.5
        
    def get_condiction(self):
        self.name = self.ui.condiction_pth.currentText()
        if self.name != '-':
            self.model_pth = os.path.join(os.getcwd(),'models', self.dict_pths[self.name])
            self.condiction_img = os.path.join(os.getcwd(),'control_images', self.dict_condiction_img[self.name])
            self.ui.condiction_status.setText(f"{self.dict_pths[self.name]} is selected")
            
            pixmap = QPixmap(self.condiction_img)
            pixmap = pixmap.scaled(81, 81, Qt.IgnoreAspectRatio)#
            scene = self.ui.condiction_image.scene()
            if scene is None:
                scene = QGraphicsScene(self)
                self.ui.condiction_image.setScene(scene)

            pixmap_item = QGraphicsPixmapItem(pixmap)            
            scene.addItem(pixmap_item)
            
            
        else:
            self.clear_frame(self.ui.condiction_image)
            self.model_pth = 'Invalid pth'
            self.ui.condiction_status.setText("Invalid pth !!")
            
    def clear_frame(self, graphics_view):
        scene = graphics_view.scene()
        if scene is not None:
            scene.clear()
            
    def set_model(self):
        if self.model is None:
            print("Setting model...")
            self.model = create_model('./models/cldm_v15.yaml').to(self.device)
            state_dict = torch.load(self.model_pth)
            self.model.load_state_dict(state_dict)
            self.ui.condiction_status.setText("Model Ready ...")
            print("Model set successfully.")
            
        else:
            print("Model is already set.")
            
    def clear_model(self):
        if self.model is not None:
            print("Clearing model...")
            del self.model
            torch.cuda.empty_cache()
            self.display_flag = False
            self.model = None
            print("Model cleared.")
            
        else:
            print("No model to clear.")
            
        self.ui.condiction_pth.setCurrentIndex(0)
        
    def display_video(self):
        #video_path = os.path.join(os.getcwd(), 'stream_src', 'videoplayback.mp4')
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)#video_path
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.HEIGHT)

        if not cap.isOpened():
            print("Can't load stream")
            return

        while True :
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = self.resize_frame(frame, target_height=self.HEIGHT, target_width=self.WIDTH)
            self.display_frame(resized_frame, self.ui.display_stream)
            
            if self.snapshot_status:
                self.display_frame(resized_frame, self.ui.display_snapshot)
                self.src_img = resized_frame
                self.snapshot_status = False
                
            QApplication.processEvents()

        cap.release()


    def display_frame(self, frame, graphics_view):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb_frame.shape
    
        view_width = graphics_view.width()
        view_height = graphics_view.height()
    
        scale_x = view_width / width
        scale_y = view_height / height
        scale = min(scale_x, scale_y)

        new_width = int(width * scale)
        new_height = int(height * scale)
    
        resized_frame = cv2.resize(rgb_frame, (new_width, new_height))
        byte_data = resized_frame.tobytes()

        q_img = QImage(byte_data, new_width, new_height, 3 * new_width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        pixmap = pixmap.scaled(481, 331, Qt.IgnoreAspectRatio)#
        scene = graphics_view.scene()
    
        if scene is None:
            scene = QGraphicsScene()
            graphics_view.setScene(scene)
            scene.clear()

        scene.addPixmap(pixmap)
        
    def resize_frame(self, img, target_height=None, target_width=None):
        h, w = img.shape[:2]

        if target_height is not None and target_width is not None:
            
            return cv2.resize(img, (target_width, target_height))
        
        elif target_height is not None:
            ratio = target_height / float(h)
            new_width = int(w * ratio)
            
            return cv2.resize(img, (new_width, target_height))
        
        elif target_width is not None:
            ratio = target_width / float(w)
            new_height = int(h * ratio)
            
            return cv2.resize(img, (target_width, new_height))
        
        else:
            return img
        
    def snapshot_flag(self):
        self.snapshot_status = True
        
    def update_ddim_steps(self, value):
        self.ui.ddim_steps.setText(f"DDIM Steps : {value}")
        self.ddim_steps = value
        
    def update_strength(self, value):
        strength = round(value / 100.0, 1)
        self.ui.condiction_strength.setText(f"Condition Strength : {strength:.1f}")
        self.strength = strength
        
    def update_scale(self, value):
        self.ui.prompt_strength.setText(f"Prompt Strength : {value}")
        self.scale = value
        
    def update_eta(self, value):
        eta = round(value / 100.0, 1)
        self.ui.eta.setText(f"Eta: {eta:.1f}")
        self.eta = eta
        
    def get_prompt(self):
        self.prompt = self.ui.main_prompt_text.toPlainText()
        self.ui.main_prompt.setText('Main Prompt : O')
        
    def clear_prompt(self):
        self.prompt = None
        self.ui.main_prompt_text.clear()
        self.ui.main_prompt.setText('Main Prompt')
        
    def get_a_prompt(self):
        self.a_prompt = self.ui.augmented_prompt_text.toPlainText()
        self.ui.augmented_prompt.setText('Augmented Prompt : O')
        
    def clear_a_prompt(self):
        self.a_prompt = None
        self.ui.augmented_prompt_text.clear()
        self.ui.augmented_prompt.setText('Augmented Prompt')
        
    def get_n_prompt(self):
        self.n_prompt = self.ui.negative_prompt_text.toPlainText()
        self.ui.negative_prompt.setText('Negative Prompt : O')
        
    def clear_n_prompt(self):
        self.n_prompt = None
        self.ui.negative_prompt_text.clear()
        self.ui.negative_prompt.setText('Negative Prompt')
        
    def load_detector(self,control_type):
        if control_type == 'Canny Edge':
            return CannyDetector()
        
        elif control_type == 'Pose':
            return OpenposeDetector()
        
        else:
            raise ValueError(f"Unsupported control_type: {control_type}")
            
    def process_image(self, input_image, model, detector, control_type,
                      prompt, a_prompt, n_prompt,
                      num_samples=1, image_resolution=512, detect_resolution=512,
                      ddim_steps=20, guess_mode=False, strength=1.0, scale=9.0,
                      seed=-1, eta=0.0):

        assert control_type in ['Canny Edge', 'Pose'], f"Unsupported control_type: {control_type}"
        
        with torch.no_grad():
            HWC_image = resize_image(HWC3(input_image), detect_resolution)

            if control_type == 'Canny Edge':
                detected_map = detector(HWC_image, 100, 200)
                
            elif control_type == 'Pose':
                detected_map, _ = detector(HWC_image)
                
            else:
                raise ValueError(f"Unknown control_type {control_type}")

            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
                seed_everything(seed)

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
            samples, _ = DDIMSampler(model).sample(ddim_steps, num_samples,
                                                   shape, cond, verbose=False, eta=eta,
                                                   unconditional_guidance_scale=scale,
                                                   unconditional_conditioning=un_cond)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()
            x_samples = np.clip(x_samples, 0, 255).astype(np.uint8)

            return x_samples[0]
        
    def get_result(self):
        detector = self.load_detector(self.name)
        result = self.process_image(self.src_img, self.model, detector, control_type=self.name, 
                                    prompt=self.prompt, a_prompt=self.a_prompt, n_prompt=self.n_prompt,
                                    num_samples=self.num_samples, image_resolution=self.image_resolution, detect_resolution=self.detect_resolution,
                                    ddim_steps=self.ddim_steps, guess_mode=self.guess_mode,
                                    strength=self.strength, scale=self.scale,
                                    seed=self.seed, eta=self.eta)
        resized_result = self.resize_frame(result, target_height=self.HEIGHT, target_width=self.WIDTH)
        self.display_frame(resized_result, self.ui.display_result)