import os
import datetime

class PathSwitcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_path": ("STRING", {"default": "FLUX/", "multiline": False}),
                "use_alt_path": ("BOOLEAN", {"default": False}),
                "date_format": ("STRING", {"default": "%date:yyyy_MM_dd%", "multiline": False}),
            },
            "optional": {
                "alt_path": ("STRING", {"default": "FLUX/backup/", "multiline": False}),
                "filename_template": ("STRING", {"default": "Skull&Worms", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "build_path"
    CATEGORY = "filename prefix handling"

    def build_path(self, base_path, use_alt_path, date_format, alt_path="FLUX/backup/", filename_template="Skull&Worms"):
        root_dir = self._clean_path(alt_path if use_alt_path else base_path)
        
        # Stały format daty (bez parsowania %date:...%)
        current_date = datetime.datetime.now().strftime("%Y_%m_%d")  # Domyślny format
        
        full_path = os.path.join(
            root_dir,
            current_date,
            self._sanitize(filename_template)
        )
        return (os.path.normpath(full_path),)

    def _clean_path(self, path):
        return os.path.normpath("".join(c for c in path.strip() if c not in '*?"<>|'))

    def _sanitize(self, name):
        return "".join(c for c in name.strip() if c not in '\\/:*?"<>|')

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

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "debug_strength_per_frame")
    FUNCTION = "colormatch_dynamic"
    DESCRIPTION = """
ColorMatch z siłą dynamicznie malejącą dla KAŻDEJ kolejnej klatki w batchu.
'falloff_duration' określa, po ilu klatkach siła ma spaść do zera.
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
        debug_strengths = {} # Słownik do przechowywania siły dla każdej klatki

        images_target_np = image_target_tensor.numpy()
        images_ref_np = image_ref_tensor.numpy()

        if image_ref_tensor.size(0) > 1 and image_ref_tensor.size(0) != batch_size:
            raise ValueError("ColorMatch: Use a single ref image or a matching batch of ref images.")

        # --- LOGIKA PRZENIESIONA DO ŚRODKA PĘTLI ---
        for i in range(batch_size):
            frame_number = i + 1 # Zaczynamy liczyć od 1, nie od 0
            strength = 0.0
            
            if frame_number <= 1:
                strength = 1.0
            elif frame_number <= falloff_duration:
                # Obliczanie siły na podstawie numeru AKTUALNEJ klatki
                normalized_frame = (frame_number - 1) / (falloff_duration - 1)
                angle = normalized_frame * (math.pi / 2)
                strength = math.cos(angle)
            
            # Zapisz siłę dla tej klatki do debugowania
            debug_strengths[f"frame_{frame_number}"] = round(strength, 4)
            print(f"Frame {frame_number}/{batch_size}, Strength: {strength:.4f}")

            # Jeśli siła jest minimalna, możemy pominąć transfer, ale wciąż blendujemy
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
                
                # Blendowanie z siłą obliczoną DLA TEJ KONKRETNEJ KLATKI
                blended_image_np = image_target_np_single + strength * (image_result_np - image_target_np_single)
            
            out.append(torch.from_numpy(blended_image_np))
            
        out_tensor = torch.stack(out, dim=0).to(torch.float32)
        out_tensor.clamp_(0, 1)
        
        debug_json_string = json.dumps(debug_strengths, indent=2)
        
        return (out_tensor, debug_json_string)
        
