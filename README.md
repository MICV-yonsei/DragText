<p align="center">
  <h1 align="center">DragText: Rethinking Text Embedding in Point-based Image Editing</h1>
  <h3 align="center">Round 1 Accepted @ WACV 2025</h3>
  <p align="center">
    <a href="https://micv-yonsei.github.io/author/gayoon-choi/"><strong>Gayoon Choi</strong></a> · 
    <a href="https://micv-yonsei.github.io/author/taejin-jeong/"><strong>Taejin Jeong</strong></a> · 
    <a href="https://micv-yonsei.github.io/author/sujung-hong/"><strong>Sujung Hong</strong></a> · 
    <a href="https://micv-yonsei.github.io/author/jaehoon-joo/"><strong>Jaehoon Joo</strong></a> · 
    <a href="https://micv-yonsei.github.io/#professor"><strong>Seong Jae Hwang</strong></a>
    <br>
    <b>Yonsei University</b>
  </p>
  <p align="center">
    <a href="https://arxiv.org/abs/2407.17843"><img alt='arXiv' src="https://img.shields.io/badge/arXiv-2306.14435-b31b1b.svg"></a>
    <a href="https://micv-yonsei.github.io/dragtext2025/"><img alt='Project Page' src="https://img.shields.io/badge/Project-Website-orange"></a>
    <!-- <a href=""><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a> -->
  </p>
  <div align="center">
    <img src="https://github.com/user-attachments/assets/f7570432-443a-4965-923d-1e4acca0e106" width="700">
  </div>
  <!--Place holder for demo video--->
  <!-- <div align="center">
    <img src="", width="700">
  </div> -->
  <br>
</p>


## 📜 Contents
* [News and Update](#news-and-update)
* [TODO](#todo)
* [Requirements](#requirements)
* [Run DragText with Gradio User Interface](#run-dragtext-with-gradio-user-interface)
* [Run DragText for Drag Bench Evaluation](#drag-bench-evaluation)
* [Citation](#citation)

<br> 

## 📢 News and Update
🚨 *DragDiffusion* and its follow-up works, including **DragText**, were developed based on `runawayml/stable-diffusion-1.5`. However, this file has been completely removed from HuggingFace and is no longer accessible. As a temporary solution, we are exploring alternatives such as `CompVis/stable-diffusion-v1-4`, `Lykon/dreamshaper-8`, and `benjamin-paine/stable-diffusion-v1-5`, following the guidelines of the diffusers communities. However, we would like to inform you in advance that these alternatives may not guarantee the same results as those presented in the original papers.

* [Sep 4th] v0.0.0 Release. 
  * Implement basic function of DragText
  * We do not support the "Editing Generated Image" tab from DragDiffusion. However, you may use the "Editing Real Image" tab to edit diffusion-generated images instead.
  * We do not support the StyleGAN2 version of FreeDrag. DragText is developed for text-conditioned diffusion models.

<br> 

## ✔️ TODO  
- [ ] Release inference code and model
- [ ] Release Gradio demo
- [ ] Open in Colab
- [ ] Enable embedding controll in User Interface

<br> 

## 🛠️ Requirements
Currently, we support four drag editing methods: *DragDiffusion*, *FreeDrag*, *DragNoise*, and *GoodDrag*. These methods are all based on *DragDiffusion* but differ in the required library versions and supported User Interfaces. **Therefore, we recommend setting up separate virtual environments and running the code independently for each method, rather than managing all four methods simultaneously.** If you encounter any issues when building the virtual environments, please refer to each method's repository for guidance.

Additionally, it is recommended to run our code on an Nvidia GPU with a Linux system. We have not tested it on other configurations. On average, DragText requires around 14GB GPU memory.

To install the required libraries, clone our repository and simply run the following commands:
```bash
git clone https://github.com/MICV-yonsei/DragText.git
```

### DragDiffusion Environment:
```bash
cd ./DragDiffusion
conda env create -f environment.yaml
conda activate dragdiff
```

### FreeDrag Environment:
```bash
cd ./FreeDrag--diffusion--version
conda env create -f environment.yaml
conda activate freedragdif
```

### DragNoise Environment:
```bash
cd ./DragNoise
conda env create -f environment.yaml
conda activate dragnoise
```

### GoodDrag Environment:
```bash
cd ./GoodDrag
conda env create -f environment.yaml
conda activate GoodDrag
```

<br>

## 🐕 Run DragText with Gradio User Interface
To start drag editing in user-interactive manner, run the following to start the gradio:

### DragDiffusion Gradio:
```bash
cd ./DragText/DragDiffusion
conda activate dragdiff
python drag_ui.py
```

### FreeDrag Gradio:
```bash
cd ./DragText/FreeDrag/FreeDrag--diffusion--version
conda activate freedragdif
python drag_ui.py
```

### DragNoise Gradio:
```bash
cd ./DragText/DragNoise
conda activate dragnoise
python drag_ui.py
```

### GoodDrag Gradio:
```bash
cd ./DragText/GoodDrag
conda activate GoodDrag
python gooddrag_ui.py
```

Although each method has differences in its interface, the basic usage is the same.

### Step 1. Train a LoRA
* Place your input image in the left-most box.
* Enter a prompt describing the image in the "prompt" field.
* Click the "Train LoRA" button to begin training the LoRA based on the input image.

### Step 2. Perform "drag" editing
* Use the left-most box to draw a mask over the areas you want to edit.
* In the middle box, select handle and target points. You can also reset all points by clicking the "Undo point" button.
* Click the "Run" button to apply the algorithm, and the edited results will appear in the right-most box.

If you're interested in the details of each individual interface, please refer to each method's repository for guidance.

<br>

## 🔍 Run DragText for Drag Bench Evaluation
**DragText** is evaluated under DragBench Dataset. We provide evaluation code for *DragDiffusion*, *FreeDrag*, *DragNoise*, and *GoodDrag* (both with and without **DragText**). 

To evaluate using DragBench, follow the steps below:

### Step 1. Download the Dataset
Download [DragBench](https://github.com/DragText/DragText/tree/main/DragDiffusion/drag_bench_evaluation) and place it in the folder `./(METHOD-FOLDER)/drag_bench_evaluation/drag_bench_data/` then unzip the files. The resulting directory structure should look like the following:
```
drag_bench_data
--- animals
------ JH_2023-09-14-1820-16
------ JH_2023-09-14-1821-23
------ JH_2023-09-14-1821-58
------ ...
--- art_work
--- building_city_view
--- ...
--- other_objects
```

### Step 2. Train LoRA
Train one LoRA on each image in `drag_bench_data` folder. We follow hyperparameters provided each method (e.g. fine-tuning steps, learning rate, rank, etc.).
```bash
python run_lora_training.py
```

### Step 3. Run Drag Editing
You can easily control whether to apply **DragText** by using the `--optimize_text` argument. For example:
```bash
# w/o DragText (Original method)
python run_drag_diffusion.py

# w/ DragText (+ Optimize text embedding)
python run_drag_diffusion.py --optimize_text --text_lr 0.004 --text_mask --text_lam 0.1 # default settings
```

### Step 4. Evaluate LPIPS, CLIP Similiarity, and Mean Distance (MD) 
Please note that executing the evaluation code for Mean Distance requires around 40GB GPU memory.
```bash
# LPIPS and CLIP similiarity
python run_eval_similarity.py

# Mean Distance
python run_eval_point_matching.py
```

For more information, check [Drag Bench Evaluation](https://github.com/DragText/DragText/tree/main/DragDiffusion/drag_bench_evaluation).

<br>

## About DragText
![image](https://github.com/user-attachments/assets/4a59c38f-ec85-47e7-b855-fda63a2689a2)
> ##### **DragText: Rethinking Text Embedding in Point-based Image Editing**
> IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025  
> Gayoon Choi, Taejin Jeong, Sujung Hong, Jaehoon Joo, Seong Jae Hwang  
> Yonsei University  

### Abstract
Point-based image editing enables accurate and flexible control through content dragging. However, the role of text embedding in the editing process has not been thoroughly investigated. A significant aspect that remains unexplored is the interaction between text and image embeddings. In this study, we show that during the progressive editing of an input image in a diffusion model, the text embedding remains constant. As the image embedding increasingly diverges from its initial state, the discrepancy between the image and text embeddings presents a significant challenge. Moreover, we found that the text prompt significantly influences the dragging process, particularly in maintaining content integrity and achieving the desired manipulation. To utilize these insights, we propose DragText, which optimizes text embedding in conjunction with the dragging process to pair with the modified image embedding. Simultaneously, we regularize the text optimization process to preserve the integrity of the original text prompt. Our approach can be seamlessly integrated with existing diffusion-based drag methods with only a few lines of code.

<br>

## Citation
If you found this code useful, please cite the following paper:  
```bibtex
@article{dragtext2024,
  title={DragText: Rethinking Text Embedding in Point-based Image Editing},
  author={Choi, Gayoon and Jeong, Taejin and Hong, Sujung and Joo, Jaehoon and Hwang, Seong Jae},
  journal={arXiv preprint arXiv:2407.17843},
  year={2024}
}
```

<br>

## Acknowledgement
This work is inspired by amazing [DragGAN](https://arxiv.org/abs/2305.10973) and [DragDiffusion](https://arxiv.org/abs/2306.14435). Also, our code is developed upon [DragDiffusion](https://github.com/Yujun-Shi/DragDiffusion), [FreeDrag](https://github.com/LPengYang/FreeDrag), [Drag Your Noise](https://github.com/haofengl/DragNoise), and [GoodDrag](https://github.com/zewei-Zhang/GoodDrag). We would also like to express our gratitude to the authors of these works and the community for their valuable contributions.

<br>

## License
Code related to the DragText algorithm is under Apache 2.0 license.

