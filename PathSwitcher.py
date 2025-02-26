import os
import datetime

class PathSwitcherNode:
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

PATH_SWITCHER_NODE_CLASS_MAPPINGS = {
    "FilePrefixSwitcher": PathSwitcherNode
}

PATH_SWITCHER_NODE_DISPLAY_NAME_MAPPINGS = {
    "FilePrefixSwitcher": "Ultimate Path Builder"
}