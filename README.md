<div align="absolute">
<h1 align="center">ESA: Energy-Based Shot Assembly Optimization for Intelligent Video Editing</h1>




## Quick Start 🚀


### Prerequisites

#### ① Prepare the dataset

1. Please place the candidate video set in the `app/lang/dataset/candidate_video` directory, and name the files in the format `segment_xxx`, where `xxx` is a  three-digit video number. Additionally, please also place the view type and camera movement record files for the candidate videos in the same directory, naming them `view_type.json` and `mv_type.json` respectively.
2. Please place the view type and camera movement record files for the reference video set in the `app/lang/dataset/reference_video` directory, naming them `view_type.json` and `mv_type.json` respectively.


#### ② Create a Python Virtual Environment

```shell
conda create -n ESA python==3.11
conda activate ESA
pip install -r requirements.txt
```


#### ③ Modify the experiment configuration

All experimental configurations are in `run_script.json`. Please pay special attention to the `video_subject`, `video_script`, `num_select`, `num_candidate_video`, and `text_list` parameters.

#### ④ Run model
Before running the following command, make sure to modify the `BASE_PATH` parameter in `app/services/task.py` according to your actual environment.

```shell
cd ESA-main
python app/services/task.py
```



