import os
import obj_segmentation
import generate_heatmap

output_dir= "output"
os.makedirs(output_dir, exist_ok=True)

# obj_segmentation.py
print("Running object segmentation...")

input_video = "input/06.mp4"
# input_video = 0
output_video = obj_segmentation.run_segmentation(input_video, output_dir)
# generate_heatmap.py
print("Running heatmap generation...")
generate_heatmap.run_heatmap_generation(output_video, output_dir)
