# video-dqn
Training and testing code for [Semantic Visual Navigation by Watching Youtube Videos](https://matthewchang.github.io/value-learning-from-videos/). Pre-processing code and more detailed readme coming soon.

## Installation

This project was developed using Python 3.7.4. Install dependencies using pip

```bash
pip install requirements.txt
```

Data for evaluation is from the [Gibson Database of Spaces](https://github.com/StanfordVL/GibsonEnv/blob/master/gibson/data/README.md). This project evaluates on environments from the `tiny` split.

This project depends on several other projects 
Habitat
Habitat-api
detectron2

## Usage

### Building Dataset

Included are scripts to download the videos in the Youtube House Tours Dataset directly from Youtube and preprocess them for Q-Learning.
```bash
cd dataset

# Downloads youtube videos
python ./extract_frames.py
```

### Training

Train a model using the Youtube House Tours Dataset

```bash
python ./train_q_network.py configs/experiments/real_data -g [GPU_ID]
```

the resulting model snapshots are saved in `configs/experiments/real_data/models/sample[SAMPLE_NUMBER].torch`

### Evaluation

Evaluate the trained model with

```bash
python ./evaluation/runner.py evaluation/config.yml -g [GPU_ID]
```

this configuration file will load the last snapshot from the training process above. To evaluate with the pretraind model run

```bash
python ./evaluation/runner.py evaluation/config_pretrained.yml -g [GPU_ID]
```

which will download the pretrained model into the project directory if it's not found. After evaluation, results can be read out using

```bash
python ./evaluation/results.py evaluation/config_pretrained.yml
```
with the appropriate config file. Evaluation videos are generated in the directory specified in the config file.

