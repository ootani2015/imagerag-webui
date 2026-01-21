[Japanese version](https://github.com/ootani2015/imagerag-webui/blob/main/README-ja.md)

<h1 align="center">ImageRAG WebUI</h1>

## Overview
ImageRAG WebUI is an enhanced image generation application built upon the original [ImageRAG](https://github.com/rotem-shalev/ImageRAG) framework. This version integrates Stable Diffusion XL (SDXL) with a Retrieval-Augmented Generation (RAG) workflow and features a user-friendly Web interface (Streamlit) for improved accessibility and functionality.

Originally designed for NVIDIA GPU (Linux) environments, this project has been optimized for Apple Silicon (M1/M2/M3 Mac). It includes support for MPS (Metal Performance Shaders) and custom memory management logic (VRAM clearing) to ensure stable execution on Mac devices.
## Setup
First, clone the repository and navigate to the project directory:
```
git clone https://github.com/ootani2015/imagerag-webui
cd imagerag-webui
```
Next, create and activate the imagerag-webui environment using Conda:
```
conda env create -f environment.yml
conda activate imagerag-webui
```
Create a datasets folder and place your image directories (e.g., Tokyo_dataset, animal_dataset) inside it as follows:
```
project/
├── datasets/
│   ├── Tokyo_dataset/
│   │   ├── bridge_01.jpg
│   │   └── bridge_02.jpg
│   └── Animal_dataset/
└── imageRAG_UI.py
```
Note: An **OpenAI API Key** is required for prompt optimization and image content evaluation.
## How to Use
### [1] Launch the App

With the imagerag-webui environment active, run the following command:
```
streamlit run imageRAG_UI.py
```


## Citation
This project is based on [ImageRAG](https://github.com/rotem-shalev/ImageRAG) with modifications and extensions developed by Rei Otani, 2025.
If you find this repository useful, please cite the ImageRAG paper:
```
@article{shalev2025imagerag,
  title={Imagerag: Dynamic image retrieval for reference-guided image generation},
  author={Shalev-Arkushin, Rotem and Gal, Rinon and Bermano, Amit H and Fried, Ohad},
  journal={arXiv preprint arXiv:2502.09411},
  year={2025}
}
```

