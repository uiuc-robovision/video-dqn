# my_project/config.py
from yacs.config import CfgNode as CN


_C = CN()

_C.PANORAMA = True
_C.SEED = 0
_C.TRAIN_ON_GROUND_TRUTH = False
_C.DATASET = "none"
_C.SUB_DATASET = "none"
_C.CLASS_LABEL = 'toilet'
_C.LOSS_CLIP = "none"
_C.ARCHITECTURE = 'basic'
_C.RANDOM_ACTIONS = False
_C.ONE_ACTION = False
_C.SEMANTIC_REWARDS = False
_C.DETECTION_REWARDS = False
_C.REMOVE_BEFORE_REWARD = False
_C.USE_INVERSE_ACTIONS = False
_C.VALUE_LEARNING = False
_C.PREVIOUS_IMAGES = False
_C.GAMMA = 0.9
_C.BOOTSTRAP = False
_C.LINEAR = False
_C.LEARNING_RATE = 1e-3
_C.NUM_STEPS = int(1e5)
_C.TARGET_UPDATE_INTERVAL = int(8e3)
_C.CHECKPOINT_INTERVAL = int(2e3)
_C.ACTION_HIDDEN_LAYERS = 1
_C.GUMBEL_TEMP = 0.1
# use confidence from detector as reward instead of binary reward
_C.CONFIDENCE_REWARD = False
_C.DISTRIBUTIONAL = False
_C.KL_BACKWARDS = False
_C.LOG_SIGMA = False
_C.VISUALIZATION_DATA_ROOT = ""


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`
