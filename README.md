# ControlNet Interactive Image Generation System

A GUI-based image generation system that captures real-time camera input and applies ControlNet-based style transformation with adjustable parameters and user-defined prompts.

The system integrates a pre-trained ControlNet model with an interactive interface, enabling users to generate stylized images from captured photos in real time.

---

## Project Overview

This project builds an end-to-end interactive image generation system that allows users to:

1. Capture images from a live camera feed  
2. Apply ControlNet-based conditional image generation  
3. Customize output style using text prompts  
4. Adjust ControlNet parameters via GUI sliders  
5. Visualize generated results instantly  

---

## System Pipeline

- Camera Input (Live Capture)  
- Image Capture Module  
- Image Preprocessing  
- ControlNet Conditional Generation  
- Prompt-based Style Control  
- Parameter Adjustment (GUI Slider)  
- Output Image Rendering  

---

## Models Used

### ControlNet Model
- Repository: https://github.com/lllyasviel/ControlNet  
- Architecture: ControlNet (based on Stable Diffusion) 
- Integration method: cloned repository used as backend module  
- Purpose: Perform conditional image generation based on input image and prompts  

---

## Model Integration

- The ControlNet model is integrated for inference-only usage  
- Pre-trained weights are loaded within the application  
- The system combines:
  - Image-based conditioning (captured photo)  
  - Text-based prompts (user-defined style)  

This design enables flexible and controllable image generation.

---

## Interactive Control Mechanism

### Parameter Adjustment
- GUI sliders allow real-time tuning of ControlNet parameters  
- Users can dynamically control generation strength and style influence  

### Prompt-based Generation
- Users can input custom prompts  
- Prompts guide the stylistic direction of generated images  

---

## Integration Strategy

This project integrates multiple components into a unified system:

- Computer Vision: Real-time camera capture  
- Generative Model: ControlNet-based image synthesis  
- User Interaction: GUI-based parameter and prompt control  

The system emphasizes real-time interaction and controllability.

---

## UI Framework

The graphical user interface provides:

- Live camera preview  
- Capture button for image acquisition  
- Prompt input field  
- Parameter sliders for ControlNet tuning  
- Generated image display  
- Interactive workflow for rapid experimentation  

---

## System Architecture

- UI Layer: GUI interface (camera + controls)  
- Capture Layer: Real-time image acquisition  
- Inference Layer: ControlNet model  
- Processing Layer: Image preprocessing and generation  
- Output Layer: Rendered image display  

---

## Key Contributions

- Designed an interactive ControlNet-based image generation system  
- Integrated real-time camera input with generative AI pipeline  
- Enabled user-controlled generation via prompts and parameter sliders  
- Built a GUI for intuitive experimentation with generative models  
- Combined computer vision and generative AI into a single workflow  

---

## Technical Challenges

- Handling real-time camera input and synchronization with model inference  
- Managing latency during image generation  
- Designing responsive UI interactions for parameter tuning  
- Integrating ControlNet inference into a standalone application  
- Balancing generation quality and performance  

---

## Requirements

### Core Environment
- Python == 3.8  
- CUDA Toolkit == 11.3   

### Deep Learning Framework
- PyTorch == 1.12.1  
- torchvision == 0.13.1  

### Generative Model Dependencies
- transformers == 4.19.2  
- pytorch-lightning == 1.5.0  
- einops == 0.3.0  
- open_clip_torch == 2.0.2  

### Computer Vision
- opencv-contrib-python == 4.3.0.36  
- albumentations == 1.3.0  
- kornia == 0.6  

### Utility Libraries
- numpy == 1.23.1  
- omegaconf == 2.1.1  
- safetensors == 0.2.7  

### Optional UI / Tools
- pyqt == 5.15.10    

### Full Dependency

For complete environment setup, refer to the original ControlNet repository:  
https://github.com/lllyasviel/ControlNet  

---

## Usage

### 1. Project Structure
```
DATA/
|-- ControlNet-main
|   |-- SDC_ui.py
|   |-- SDC_contain.py
|   |-- SDC_start.py
```
---

### 2. Run the application
```
cd code
python SDC_start.py
```
---

### 2. Workflow

1. Open the application  
2. Activate camera preview  
3. Capture an image  
4. Enter a prompt describing desired style  
5. Adjust ControlNet parameters using sliders  
6. Generate stylized image  
7. View and iterate results  

---

## Screenshot
![image](https://github.com/vGuNfnxXLJ/ControlNet-Interactive-Image-Generation-System/blob/main/RealControlNet_01.PNG)

![image](https://github.com/vGuNfnxXLJ/ControlNet-Interactive-Image-Generation-System/blob/main/RealControlNet_02.PNG)
---

## External Acknowledgements

This project is based on the following open-source work:

- ControlNet: https://github.com/lllyasviel/ControlNet  

All credits belong to the original authors.

---

## Notes

- This project is for research and experimental purposes  
- Performance depends on hardware and model configuration  
- Real-time performance may vary depending on system resources
