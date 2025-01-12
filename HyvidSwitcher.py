import nodes

class HyvidSwitcher:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 640, "min": 64, "max": 1280, "step": 16}),
                "height": ("INT", {"default": 480, "min": 64, "max": 1280, "step": 16}),
                "frame_count": ("INT", {"default": 49, "min": 1, "max": 1024, "step": 4}),
                "variables": (["User Defined", "From Input"],),
            },
            "optional": {
                "input_width": ("INT", {"forceInput": True}),
                "input_height": ("INT", {"forceInput": True}),
                "input_frame_count": ("INT", {"default": 49, "min": 1, "max": 1024, "step": 4, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "frame_count")

    FUNCTION = "variables_groups"

    CATEGORY = "Custom Nodes"
    DESCRIPTION = "Simple switch between set and received parameters adapted to HyVideo requirements"

    def adjust_frame_count_group_a(self, frame_count):
        # Zaokrągla do najbliższej wartości spełniającej warunek 1 + 4n
        n = (frame_count - 1) // 4
        adjusted_count = 1 + 4 * n
        if abs(frame_count - adjusted_count) > abs(frame_count - (adjusted_count + 4)):
            adjusted_count += 4
        return adjusted_count

    def adjust_frame_count_group_b(self, frame_count):
        # Zaokrągla do największej mniejszej lub równej wartości spełniającej warunek 1 + 4n
        if frame_count <= 1:
            return 1
        else:
            n = (frame_count - 1) // 4
            adjusted_count = 1 + 4 * n
            return adjusted_count

    def variables_groups(self, width, height, frame_count, variables, **kwargs):
        if variables == "User Defined":
            adjusted_frame_count = self.adjust_frame_count_group_a(frame_count)
            return (width, height, adjusted_frame_count)
        elif variables == "From Input":
            input_width = kwargs.get("input_width")
            input_height = kwargs.get("input_height")
            input_frame_count = kwargs.get("input_frame_count")

            if input_width is None or input_height is None or input_frame_count is None:
                raise ValueError(f"Input values are not fully provided: width={input_width}, height={input_height}, frame_count={input_frame_count}")
            
            adjusted_input_frame_count = self.adjust_frame_count_group_b(input_frame_count)
            return (input_width, input_height, adjusted_input_frame_count)
        else:
            raise ValueError(f"Unknown variable source: {variables}")

# Definicja NODE_CLASS dla ComfyUI
NODE_CLASS_MAPPINGS = {
    "HyvidSwitcher": HyvidSwitcher,
}

# Dodanie definicji NODE_DISPLAY_NAME_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = {
    "HyvidSwitcher": "HyVid Switcher",
}