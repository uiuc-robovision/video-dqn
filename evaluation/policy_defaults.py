# my_project/config.py
from yacs.config import CfgNode as CN
from defaults import get_cfg_defaults

_C = CN()
_C.INHERIT = ""

_C.SCORE = 'geodesic'
_C.DATASET = 'val'
_C.SLAM = False
_C.STOP = False
_C.MODEL_CONFIG_LOCATION = ""
_C.MODEL_NAME = ""
_C.ACT_ON_Q = False
_C.Q_STOCHASTIC = False
_C.BEHAVIOR_CLONING = False
_C.BEHAVIOR_PANORAMA = False
_C.BEHAVIOR_REAL = False
_C.BEHAVIOR_FINETUNE = False
_C.BEHAVIOR_LSTM = True
_C.RESULT_LOCATION = 'navigation_results'
_C.VIDEO_LOCATION = 'navigation_videos'
_C.CHASE_DETECTOR = False
_C.COMBINE_DETECTOR = False
_C.CONFIDENCE_THRESHOLD = 0.5
_C.SEED = 0
_C.STAIRS = False
_C.MODEL_NUMBER = 300000
_C.FORWARD_SCORE = False
_C.PREVIOUS_IMAGES_REPLICATE = False
_C.PREVIOUS_IMAGES_ROTATE = False
_C.BEHAVIOR_NONEG = False
_C.BEHAVIOR_MASK = False
_C.BEHAVIOR_LOG = False
_C.HABITAT_POLICY= False
_C.HABITAT_CONFIG_PATH= ""
_C.HABITAT_MODEL_NAME= "noname"
_C.HABITAT_FRAMES=0.0
_C.HABITAT_CHECKPOINT=0
_C.HABITAT_LOG=False
_C.HABITAT_BC_RL=False

_C.CONSISTENCY_WEIGHT = 0.0
_C.BACKTRACK_REJECTION = False
_C.TOTAL_RANDOM = False
_C.FORWARD_IMAGES = False
_C.FORWARD_IMAGE_STEPS = 4
_C.HALLUCINATE = False
_C.SINGLE_MODEL_PANORAMA = False


def name_from_config(config):
    if config.TOTAL_RANDOM:
        name = f'total_random'
    elif config.HABITAT_POLICY:
        name = f'habitat_{config.HABITAT_MODEL_NAME}'
        if config.HABITAT_CHECKPOINT != 0:
            name += f'_{config.HABITAT_CHECKPOINT}'
        else:
            name += f'_frames{int(config.HABITAT_FRAMES)}'
        if config.HABITAT_LOG: name += '_log'
    elif config.ACT_ON_Q:
        name = f'actonq_{config.MODEL_NAME}'
        if config.Q_STOCHASTIC: name += "_stochastic"
    elif config.BEHAVIOR_CLONING:
        name = 'behavior_stop' if config.STOP else 'behavior'
        if config.BEHAVIOR_LOG: name += "_log"
        name += "_panorama" if config.BEHAVIOR_PANORAMA else "_nopanorama"
        if config.BEHAVIOR_REAL: name += "_real"
        if config.BEHAVIOR_FINETUNE: name += "_finetune"
        if config.BEHAVIOR_NONEG: name += "_noneg"
        if config.BEHAVIOR_MASK: name += "_mask"
    else:
        name = config.MODEL_NAME if config.SCORE == 'model' else config.SCORE
        name += '_log' if config.STOP else "_spl"
        if config.SLAM: name += '_slam'
        if config.BACKTRACK_REJECTION: name += '_rejection'
        if config.CHASE_DETECTOR: name += '_chase'
        if config.FORWARD_SCORE: name += '_forward'
        if config.PREVIOUS_IMAGES_REPLICATE: name += '_replicate'
        if config.PREVIOUS_IMAGES_ROTATE: name += '_prev_rotate'
        if config.FORWARD_IMAGES: name += '_forward_images'
        if config.FORWARD_IMAGE_STEPS != 4: name += f'_fis{config.FORWARD_IMAGE_STEPS}'
        if config.HALLUCINATE: name += '_hallucinate'
        if config.SINGLE_MODEL_PANORAMA: name += '_single_panorama'
        if config.COMBINE_DETECTOR: 
            name += f'_combined{config.CONFIDENCE_THRESHOLD}'
        if config.CONSISTENCY_WEIGHT != 0:
            name += f'_consistency{config.CONSISTENCY_WEIGHT}'
        if config.MODEL_NUMBER != _C.MODEL_NUMBER:
            name += f'_model{config.MODEL_NUMBER}'
    if config.SEED != 0: name += f'_seed{config.SEED}'
    if config.DATASET != 'val': name += f'_{config.DATASET}'
    if config.STAIRS:  name += f'_with_stairs'

    return name


# def get_config(config_paths = None, opts = None):
# r"""Create a unified config with default values overwritten by values from
# `config_paths` and overwritten by options from `opts`.
# Args:
# config_paths: List of config paths or string that contains comma
# separated list of config paths.
# opts: Config options (keys, values) in a list (e.g., passed from
# command line into the config. For example, `opts = ['FOO.BAR',
# 0.5]`. Argument can be used for parameter sweeping or quick tests.
# """
# config = _C.clone()
# CONFIG_FILE_SEPARATOR = ','
# if config_paths:
# if isinstance(config_paths, str):
# if CONFIG_FILE_SEPARATOR in config_paths:
# config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
# else:
# config_paths = [config_paths]

# for config_path in config_paths:
# config.merge_from_file(config_path)
# if len(config.MODEL_CONFIG_LOCATION) > 0:
# cfg = get_cfg_defaults()
# cfg.merge_from_file(config.MODEL_CONFIG_LOCATION+"/config.yml")
# config.MODEL_CONFIG = cfg
# config.freeze()
# return config


def load_file(file_loc):
    cfg = _C.clone()
    cfg.merge_from_file(file_loc)
    if cfg.INHERIT != '':
        base_cfg = load_file(cfg.INHERIT)
        base_cfg.merge_from_file(file_loc)
        cfg = base_cfg

    # special sub config
    if len(cfg.MODEL_CONFIG_LOCATION) > 0:
        sub_cfg = get_cfg_defaults()
        sub_cfg.merge_from_file(cfg.MODEL_CONFIG_LOCATION + "/config.yml")
        cfg.defrost()
        cfg.MODEL_CONFIG = sub_cfg

    cfg.freeze()
    return cfg
