## Metric Point Cloud Creation from Single Images using Depth Anything Model

This project leverages the Depth Anything model to create metric point clouds from single images. This repository includes all necessary scripts and files to perform depth estimation and generate accurate 3D point clouds with real-world measurements.

### Requirements

------------

To install the required packages, run:
`pip install -r requirements.txt`

### Installation

------------

1. Clone this repository:
`git clone https://github.com/bohdanvodianyk/image-to-pcd.git`
`cd your-repo-name`
2. Install the necessary Python packages:
`pip install -r requirements.txt`
3. Download the model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1LJRnpOhNuzZXlVE0oGzzUb7ZiXeF6f-8?usp=sharing "Google Drive") and place them in the appropriate directory within the project.

### Usage

------------

#### Calibration
Before generating point clouds, calibrate your camera using the `calibration-camera.py` script. Ensure you have a chessboard pattern printed for the calibration process.

#### Depth Estimation to Point Cloud
To convert depth maps into metric point clouds, use the `depth_to_pointcloud.py` script. Ensure your input image is correctly formatted and accessible.

#### HEIC to PNG Conversion
If your input images are in HEIC format, convert them to PNG using the `heic2png.py` script.

#### Web UI
Start the local browser UI from the repository root:

```bash
python app.py
```

Open `http://127.0.0.1:5000`, select at least two JPG room photos, and follow the upload, point-cloud conversion, and registration steps. After registration, inspect the merged point cloud in the interactive home-page preview: drag to orbit and scroll to zoom. The merged point cloud and extracted floor-plan point cloud are also available to download. The UI requires the dependencies in `requirements.txt`, the camera calibration file, and the indoor model checkpoint.

### Model Checkpoints

------------

Model checkpoints necessary for depth estimation can be downloaded from the following Google Drive link:

[Google Drive - Model Checkpoints](https://drive.google.com/drive/folders/1LJRnpOhNuzZXlVE0oGzzUb7ZiXeF6f-8?usp=sharing "Google Drive - Model Checkpoints")

Download and place these checkpoints in the appropriate directory within your project to ensure the model functions correctly.

### Acknowledgements

------------

Special thanks to the developers of the **Depth Anything** model and all contributors who made this project possible. Your work in computer vision and deep learning is greatly appreciated.
