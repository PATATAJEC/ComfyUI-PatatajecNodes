Hopefully I will be able to create some useful nodes for ComfyUI. I just started making these, using LLMs for coding purposes, with no previous coding experience. However, I can see myself learning more and more. 

### HyvidSwitcher Node

The HyvidSwitcher node is a custom node designed to switch between user-defined parameters and input-based parameters. It helps with connecting chunks of workflow like v2v with i2v and was made specifically for Hunyuan Video purposes, as it formats resolutions and frame count for that specific model. It allows users to dynamically choose whether to use predefined settings or values provided as inputs with the help of a simple switch which is just a selection of modes:
- User Defined: Uses predefined values set by the user.
- From Input: Uses values provided as inputs to the node.

It's simple. You can define your width, height and frame count. It changes these values automatically to be compatible with the Hunyuan Video model. It makes resolution capped between 64 and 1280, and divisible by 16. Also frame count is Hunyuan-specific, which means it allows only frame counts of 1+4n frames (1,5,9...33...129 etc.). It's then output formatted data (width, height, frame count). 
