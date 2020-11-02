# Semantic Visual Navigation by Watching Youtube Videos
Training and testing code for [Semantic Visual Navigation by Watching Youtube Videos](https://matthewchang.github.io/value-learning-from-videos/), appearing in NeurIPS 2020.

<div align="center">
  <img src="https://github.com/MatthewChang/value-learning-from-videos/blob/master/video-dqn-website_files/gif1.gif"/>
</div>

## Installation

This project was developed using Python 3.7.4. Install dependencies using pip

```bash
pip install -r requirements.txt
```

Additionally this project depends on [habitat-sim v0.1.4](https://github.com/facebookresearch/habitat-sim/tree/v0.1.4), [habitat-api v0.1.3](https://github.com/facebookresearch/habitat-lab/tree/v0.1.3) (now renamed to [habitat-lab](https://github.com/facebookresearch/habitat-lab)), and [detectron2 v0.1](https://github.com/facebookresearch/detectron2/tree/v0.1). The installation instructions for these projects can be found on their respective webpages linked above.

Data for evaluation is from the [Gibson Database of Spaces](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md). This project evaluates on environments from the `tiny` split. Using object annotations from [3D Scene Graph](https://github.com/StanfordVL/3DSceneGraph)

Once the gibson data has been downloaded (using the "Gibson Database for Habitat-sim" link from the site above). You will need to proved the path to that data (the folder containing navmeshes and manifest file) as an environment variable at test time (see below). You also need the For our experiments we regenerated the navmeshes using an agent height of 1.25 to allow the scenes to be traversable through some low doorways and a max climb value of 0.05 to disallow climbing stairs. A description of the modifications made (which require editing the source files of habitat-sim) and the script used for regenerating the meshes can be found in `regenerate_navmeshes.rb`. This step may not be necessary for later versions of habitat-sim as they seem to have added functionality to programatically [recompute the navmeshes](https://github.com/facebookresearch/habitat-sim/pull/333) when the agent parameters change. However, this code was not tested with these versions of habitat-sim.


## Usage

### Building Dataset

Included are scripts to download the videos in the Youtube House Tours Dataset directly from Youtube and preprocess them for Q-Learning.
```bash
# Downloads youtube videos
python dataset/download_videos.py
# Splits out frames
python dataset/extract_frames.py --dump
# Find frames with people and outdoor scenes
python dataset/extract_frames.py
# Run object detector
python dataset/detect_real_videos.py
# Build the dataset file
python dataset/process_episodes_real.py
```
the above scripts produce a file `dataset/data.feather` which contains the Q-Learning quadruplets for training. 

### Training

Train a model using the Youtube House Tours Dataset

```bash
python ./train_q_network.py configs/experiments/real_data -g [GPU_ID]
```

the resulting model snapshots are saved in `configs/experiments/real_data/models/sample[SAMPLE_NUMBER].torch`

### Evaluation

To evaluate you must have two environment variables set, one to reference the gibson meshes, one for the 3D scene graph annotations location. Evaluate the trained model with

```bash
SCENE_GRAPH_LOCATION_TINY=[3d_scene_graph_tiny_annotations_location] GIBSON_LOCATION=[gibson_path] python ./evaluation/runner.py evaluation/config.yml -g [GPU_ID]
```

this configuration file will load the last snapshot from the training process above. To evaluate with the pretraind model run

```bash
SCENE_GRAPH_LOCATION_TINY=[3d_scene_graph_tiny_annotations_location] GIBSON_LOCATION=[gibson_path] python ./evaluation/runner.py evaluation/config_pretrained.yml -g [GPU_ID]
```

which will download the pretrained model into the project directory if it's not found. After evaluation, results can be read out using

```bash
python ./evaluation/results.py evaluation/config_pretrained.yml
```
with the appropriate config file. Evaluation videos are generated in the directory specified in the config file.
