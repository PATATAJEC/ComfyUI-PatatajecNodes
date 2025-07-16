import os
import datetime

class PathTool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "ComfyUI", "multiline": False}),
                "base_path": ("STRING", {"default": "Project_01/", "multiline": False}),
                "alt_path": ("STRING", {"default": "Project_01/Alternatives/", "multiline": False}),
                "use_alt_path": ("BOOLEAN", {"default": False}),
                "date_format": ("STRING", {"default": "%Y_%m_%d", "multiline": False}),
                "use_date_folder": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "build_path"
    CATEGORY = "Filename & Path Manager"
    DESCRIPTION = """
### Path Tool

This node dynamically generates a filename prefix for saving files into an organized folder structure.

What it does:
* It constructs a path in the format: `[root_directory]/[current_date]/[filename]`.
* You can choose between a primary `base_path` and an `alt_path` using a boolean toggle.
* It automatically creates a subfolder named with the current date in YYYY_MM_DD format (e.g., 2023_10_27).
* It sanitizes the path and filename to remove any invalid characters, ensuring a safe output.
* The final string is intended to be connected to the `filename_prefix input` of a "Save Image" or similar node. 
"""

    def build_path(self, filename_template, base_path, alt_path, use_alt_path, date_format, use_date_folder):
        root_dir = self._clean_path(alt_path if use_alt_path else base_path)
         path_components = [root_dir]
        
        if use_date_folder:
            current_date_str = datetime.datetime.now().strftime(date_format)
            path_components.append(current_date_str)
            
        sanitized_filename = self._sanitize(filename_template)
        path_components.append(sanitized_filename)
        
        full_path = os.path.join(*path_components)
        
        return (os.path.normpath(full_path),)

    def _clean_path(self, path):
        return os.path.normpath("".join(c for c in path.strip() if c not in '*?"<>|'))

    def _sanitize(self, name):
        return "".join(c for c in name.strip() if c not in '\\/:*?"<>|')

import math
import torch
import json 

class ColorMatchFalloff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_ref": ("IMAGE",),
                "image_target": ("IMAGE",),
                "method": (
                    ['mkl', 'hm', 'reinhard', 'mvgd', 'hm-mvgd-hm', 'hm-mkl-hm'], 
                    {"default": 'mkl'}
                ),
                "falloff_duration": ("INT", {
                    "default": 50, 
                    "min": 2, 
                    "max": 1000, 
                    "display": "number"
                }),
            },
        }
    
    CATEGORY = "KJNodes/image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "colormatch_dynamic"
    DESCRIPTION = """
Code is modified version of Kijai's ComfyUI-KJNodes (https://github.com/kijai/ComfyUI-KJNodes).
Color Match Falloff node smoothly fades the color correction effect over a sequence of frames (falloff_duration). It uses an Ease-In-Out curve (from strength 1 to 0) to mitigate color flickering, making it ideal for seamlessly stitching together video clips generated in separate batches (last frame to first frame approach) using WAN2.1 model
"""
    
    def colormatch_dynamic(self, image_ref, image_target, method, falloff_duration=50):
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            raise Exception("Can't import color-matcher. Pip install color-matcher")
        
        cm = ColorMatcher()
        image_ref_tensor = image_ref.cpu()
        image_target_tensor = image_target.cpu()
        batch_size = image_target_tensor.size(0)
        
        out = []

        images_target_np = image_target_tensor.numpy()
        images_ref_np = image_ref_tensor.numpy()

        if image_ref_tensor.size(0) > 1 and image_ref_tensor.size(0) != batch_size:
            raise ValueError("ColorMatch: Use a single ref image or a matching batch of ref images.")

        for i in range(batch_size):
            frame_number = i + 1 
            strength = 0.0
            
            if frame_number <= 1:
                strength = 1.0
            elif frame_number <= falloff_duration:
                normalized_frame = (frame_number - 1) / (falloff_duration - 1)
                angle = normalized_frame * (math.pi / 2)
                strength = math.cos(angle)

            if strength < 0.0001:
                blended_image_np = images_target_np[i]
            else:
                image_target_np_single = images_target_np[i]
                image_ref_np_single = images_ref_np[0] if image_ref_tensor.size(0) == 1 else images_ref_np[i]
                
                try:
                    image_result_np = cm.transfer(src=image_target_np_single, ref=image_ref_np_single, method=method)
                except Exception as e:
                    print(f"Error during color transfer on frame {frame_number}: {e}")
                    image_result_np = image_target_np_single
                
                blended_image_np = image_target_np_single + strength * (image_result_np - image_target_np_single)
            
            out.append(torch.from_numpy(blended_image_np))
            
        out_tensor = torch.stack(out, dim=0).to(torch.float32)
        out_tensor.clamp_(0, 1)
     
        return (out_tensor,)
        
