# DragText: Rethinking Text Embedding in Point-based Image Editing
#### Early Accepted @ WACV 2025 
We will release the code soon. Stay tuned!

[[Project Page]](https://micv-yonsei.github.io/dragtext2025/) [[arXiv]](https://arxiv.org/abs/2407.17843)
<be>

![image](https://github.com/user-attachments/assets/4a59c38f-ec85-47e7-b855-fda63a2689a2)
> ##### **DragText: Rethinking Text Embedding in Point-based Image Editing**
> IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2025  
> Gayoon Choi, Taejin Jeong, Sujung Hong, Jaehoon Joo, Seong Jae Hwang  
> Yonsei University
### Abstract
Point-based image editing enables accurate and flexible control through content dragging. However, the role of text embedding in the editing process has not been thoroughly investigated. A significant aspect that remains unexplored is the interaction between text and image embeddings. In this study, we show that during the progressive editing of an input image in a diffusion model, the text embedding remains constant. As the image embedding increasingly diverges from its initial state, the discrepancy between the image and text embeddings presents a significant challenge. Moreover, we found that the text prompt significantly influences the dragging process, particularly in maintaining content integrity and achieving the desired manipulation. To utilize these insights, we propose DragText, which optimizes text embedding in conjunction with the dragging process to pair with the modified image embedding. Simultaneously, we regularize the text optimization process to preserve the integrity of the original text prompt. Our approach can be seamlessly integrated with existing diffusion-based drag methods with only a few lines of code.

## Citation
If you found this code useful, please cite the following paper:  
```
@article{dragtext2024,
  title={DragText: Rethinking Text Embedding in Point-based Image Editing},
  author={Choi, Gayoon and Jeong, Taejin and Hong, Sujung and Joo, Jaehoon and Hwang, Seong Jae},
  journal={arXiv preprint arXiv:2407.17843},
  year={2024}
}
```
