## Getting Started
To get started with this project, follow these steps:

### Prerequisites
Ensure you have the following installed:
- Python 3. x
- OpenCV
- Pillow
- NumPy
- Ultralytics YOLO

### Installation
Clone the repository:
```sh
git clone https://github.com/yourusername/object-segmentation-tracking-heatmap.git
cd object-segmentation-tracking-heatmap
```

### Install the required dependencies:
Copy & paste the code into Terminal
```sh
pip install -r requirements.txt
```
### Usage

Prepare Your Video: Place your video files in the input directory.

Run the Pipeline: Execute the main script to process your videos.
Copy & paste the code into Terminal
```sh
python run_full_pipeline.py --input input/video.mp4 --output output/
```
View Results: Check the output directory for processed videos, object tracks, and heatmaps.

### Project Structure
input/: Directory for input videos.

output/: Directory for output videos and visualizations.

scripts/: Contains Python scripts for segmentation, tracking, and heatmap generation.

notebooks/: Jupyter notebooks for model evaluation and analysis.
