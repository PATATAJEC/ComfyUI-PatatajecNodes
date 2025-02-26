import logging  # Dodano import modułu logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)  # Można zmienić na logging.WARNING, aby wyłączyć debugowanie

class VideoCounter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_count": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),  # Liczba wejść
            },
            "optional": {
                # Dynamiczne wejścia będą dodane przez przycisk "Update Inputs"
            }
        }

    RETURN_TYPES = ("LIST",)  # Zwraca listę w formacie {"v1": [[wartość]], "v2": [[wartość]], ...}
    RETURN_NAMES = ("video_frame_list",)  # Zmieniono nazwę wyjścia na video_frame_list
    FUNCTION = "update_inputs"
    CATEGORY = "Video Processing"
    DESCRIPTION = """
Dynamiczny węzeł do przydzielania wejść.  
Możesz ustawić liczbę wejść za pomocą **input_count** i kliknąć "Update".
Wszystkie wejścia są dynamiczne.
"""

    def __init__(self):
        # Inicjalizacja liczby wejść
        self.input_count = 1

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # Ta metoda jest wymagana, aby ComfyUI wiedział, kiedy węzeł się zmienia
        return float("nan")

    def update_inputs(self, input_count, **kwargs):
        logging.info("Debug: Updating inputs...")  # Zmieniono print na logging.info

        # Zbierz wartości z dynamicznych wejść
        video_frame_list = {}
        for i in range(1, input_count + 1):
            input_name = f"v{i}_fcount"
            if input_name in kwargs:
                video_frame_list[f"v{i}"] = [[kwargs[input_name]]]

        logging.info("Debug: Generated video_frame_list: %s", video_frame_list)  # Zmieniono print na logging.info

        # Zwróć listę video_frame_list
        return (video_frame_list,)

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # Ta metoda może być używana do walidacji wejść
        return True

# Zmiana nazw zmiennych mapowań
VIDEO_COUNTER_NODE_CLASS_MAPPINGS = {
    "VideoCounter": VideoCounter,
}

VIDEO_COUNTER_NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCounter": "Video Counter",
}