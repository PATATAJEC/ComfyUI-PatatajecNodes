import torch

class ImageSequenceFromBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "sequence": ("STRING", { "default": "0,1,2,3,4,0,1,0,1,2,3,20", "multiline": False }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image, sequence):
        try:
            # Parse the sequence string into a list of integers
            indices = [int(idx.strip()) for idx in sequence.split(",")]
        except ValueError:
            raise ValueError("Invalid sequence format. Please provide a comma-separated list of integers.")

        # Ensure all indices are within the valid range
        max_index = image.shape[0] - 1
        indices = [min(max(idx, 0), max_index) for idx in indices]

        # Extract the images based on the sequence
        output_images = [image[idx].unsqueeze(0) for idx in indices]
        output_images = torch.cat(output_images, dim=0)

        return (output_images,)

# Zmiana nazw zmiennych mapowań
IMAGE_SEQUENCE_FROM_BATCH_NODE_CLASS_MAPPINGS = {
    "ImageSequenceFromBatch": ImageSequenceFromBatch,
}

IMAGE_SEQUENCE_FROM_BATCH_NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageSequenceFromBatch": "🔧 Image Sequence From Batch",
}