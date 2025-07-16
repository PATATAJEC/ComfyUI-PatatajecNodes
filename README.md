Hopefully I will be able to create some useful nodes for ComfyUI. I just started making these, using LLMs for coding purposes, with no previous coding experience. However, I can see myself learning more and more. 

### Path Tool

This node dynamically generates a filename prefix for saving files into an organized folder structure.

What it does:
* It constructs a path in the format: `[root_directory]/[current_date]/[filename]`.
* You can choose between a primary `base_path` and an `alt_path` using a boolean toggle.
* It automatically creates a subfolder named with the current date in YYYY_MM_DD format (e.g., 2023_10_27).
* It sanitizes the path and filename to remove any invalid characters, ensuring a safe output.
* The final string is intended to be connected to the `filename_prefix input` of a "Save Image" or similar node.
