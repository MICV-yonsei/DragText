# *************************************************************************
# Copyright (2023) Bytedance Inc.
#
# Copyright (2023) DragDiffusion Authors 
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at 
#
#     http://www.apache.org/licenses/LICENSE-2.0 
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
# See the License for the specific language governing permissions and 
# limitations under the License. 
# *************************************************************************

# run results of DragDiffusion
import argparse
import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import PIL
from PIL import Image

from copy import deepcopy
from einops import rearrange
from types import SimpleNamespace

from diffusers import DDIMScheduler, AutoencoderKL
from torchvision.utils import save_image
from pytorch_lightning import seed_everything

import sys
sys.path.insert(0, '../')
from pipeline import GoodDragger

from utils.ui_utils import get_original_points
from utils.attn_utils import register_attention_editor_diffusers, MutualSelfAttentionControl


def preprocess_image(image,
                     device):
    image = torch.from_numpy(image).float() / 127.5 - 1 # [-1, 1]
    image = rearrange(image, "h w c -> 1 c h w")
    image = image.to(device)
    return image

# copy the run_drag function to here
def run_drag(source_image,
             image_with_clicks,
             mask,
             prompt,
             points,
             inversion_strength,
             lam,
             text_lam,
             latent_lr,
             text_lr,
             feature_idx,
             model_path,
             vae_path,
             lora_path,
             optimize_text, 
             text_mask, 
             # save_dir="./results"
    ):
    # initialize model
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    height, width = source_image.shape[:2]
    
    n_inference_step = 50
    guidance_scale = 1.0
    drag_end_step = 7
    track_per_step = 10

    r1 = 4
    r2 = 12
    d = 4
    max_drag_per_track = 3
    max_track_no_change = 5

    drag_loss_threshold = 0
    compare_mode = False
    once_drag = False
    return_intermediate_images = False

    seed = 42

    dragger = GoodDragger(device, model_path, prompt, height, width, inversion_strength, r1, r2, d,
                          drag_end_step, track_per_step, lam, latent_lr,
                          n_inference_step, guidance_scale, feature_idx, compare_mode, vae_path, lora_path, seed,
                          max_drag_per_track, drag_loss_threshold, once_drag, max_track_no_change,
                          optimize_text, text_lr, text_mask, text_lam)

    source_image = preprocess_image(source_image, device)
    image_with_clicks = preprocess_image(image_with_clicks, device)

    try:    
        # inference the synthesized image
        gen_image, intermediate_features, new_points_handle, intermediate_images = \
            dragger.good_drag(source_image, points, mask, return_intermediate_images)
            
        # resize gen_image into the size of source_image
        # we do this because shape of gen_image will be rounded to multipliers of 8
        new_points_handle = get_original_points(new_points_handle, height, width, dragger.sup_res_w, dragger.sup_res_h)

        gen_image = F.interpolate(gen_image, (height, width), mode='bilinear')
    except IndexError:
        print("Failed to drag the image {}".format(sample_name))
        gen_image = source_image
        
    # save the original image, user editing instructions, synthesized image
    save_result = torch.cat([
        source_image * 0.5 + 0.5,
        torch.ones((1,3,height,25)).cuda(),
        image_with_clicks * 0.5 + 0.5,
        torch.ones((1,3,height,25)).cuda(),
        gen_image[0:1]
    ], dim=-1)

    out_image = gen_image.cpu().permute(0, 2, 3, 1).numpy()[0]
    out_image = (out_image * 255).astype(np.uint8)

    # new_points = []
    # for i in range(len(new_points_handle)):
    #     new_cur_handle_points = new_points_handle[i].numpy().tolist()
    #     new_cur_handle_points = [int(point) for point in new_cur_handle_points]
    #     new_points.append(new_cur_handle_points)
    #     new_points.append(points[i * 2 + 1])

    return out_image, save_result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="setting arguments")
    parser.add_argument('--lora_steps', type=int, default=70, help='number of lora fine-tuning steps')
    parser.add_argument('--inv_strength', type=float, default=0.75, help='inversion strength')
    parser.add_argument('--latent_lr', type=float, default=0.02, help='latent learning rate')
    parser.add_argument('--unet_feature_idx', type=int, default=3, help='feature idx of unet features')
    parser.add_argument('--prefix', type=str, help='result path')
    parser.add_argument('--optimize_text', action="store_true", help='update text embeddings')
    parser.add_argument('--text_lr', type=float, default=0.004, help='text embeddings learning rate')
    parser.add_argument('--text_mask', action="store_true", help='appling text mask')
    parser.add_argument('--text_lam', type=float, default=0.1, help='regularization strength for text mask')
    args = parser.parse_args()

    all_category = [
        'art_work',
        'land_scape',
        'building_city_view',
        'building_countryside_view',
        'animals',
        'human_head',
        'human_upper_body',
        'human_full_body',
        'interior_design',
        'other_objects',
    ]

    # assume root_dir and lora_dir are valid directory
    root_dir = 'drag_bench_data'
    lora_dir = 'drag_bench_lora'
    result_dir = 'results' + \
        '/' + str(args.prefix) + \
        '_' + 'drag_diffusion_res' + \
        '_' + str(args.lora_steps) + \
        '_' + str(args.inv_strength) + \
        '_' + str(args.latent_lr) + \
        '_' + str(args.unet_feature_idx)

    # mkdir if necessary
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
        for cat in all_category:
            os.mkdir(os.path.join(result_dir,cat))

    for cat in all_category:
        file_dir = os.path.join(root_dir, cat)
        for sample_name in os.listdir(file_dir):
            if sample_name == '.DS_Store':
                continue
            sample_path = os.path.join(file_dir, sample_name)

            # read image file
            source_image = Image.open(os.path.join(sample_path, 'original_image.png'))
            source_image = np.array(source_image)
            image_with_clicks = Image.open(os.path.join(sample_path, 'user_drag.png'))
            image_with_clicks = np.array(image_with_clicks)

            # load meta data
            with open(os.path.join(sample_path, 'meta_data.pkl'), 'rb') as f:
                meta_data = pickle.load(f)
            prompt = meta_data['prompt']
            mask = meta_data['mask']
            points = meta_data['points']

            # load lora
            lora_path = os.path.join(lora_dir, cat, sample_name, str(args.lora_steps))
            print("applying lora: " + lora_path)
            
            out_image, save_result = run_drag(
                source_image,
                image_with_clicks,
                mask,
                prompt,
                points,
                inversion_strength=args.inv_strength,
                lam=0.1,
                text_lam=args.text_lam, 
                latent_lr=args.latent_lr,
                text_lr=args.text_lr,
                feature_idx=args.unet_feature_idx,
                model_path="runwayml/stable-diffusion-v1-5",
                vae_path="stabilityai/sd-vae-ft-mse",
                lora_path=lora_path,
                optimize_text=args.optimize_text,
                text_mask=args.text_mask, 
                # save_dir="./results"
            )
            save_dir = os.path.join(result_dir, cat, sample_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            Image.fromarray(out_image).save(os.path.join(save_dir, 'dragged_image.png'))
            save_image(save_result, os.path.join(save_dir, 'compare_images.png'))
            torch.cuda.empty_cache()
