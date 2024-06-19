# Vision Language Mapping

This repo uses clipseg for semantic segmentation and create a costmap using lidar pointclouds

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install dependencies.

```bash
pip install -r requirements.txt
```

## Usage
To run segmentation from a folder containing images.

```python
python run/run_folder.py --image_folder /path/to/images --prompts "prompt_1" "prompt_2" ... "prompt_n" --output_folder /path/to/output
```
To run segmentation from a rosbag.

```python
python run/run_bag.py --bag_file_path /path/to/rosbag --prompts "prompt_1" "prompt_2" ... "prompt_n" --topic <image_topic>
```
