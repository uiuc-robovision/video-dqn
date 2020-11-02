from habitat.config import Config as CN  # type: ignore
from habitat.config.default import _C, CONFIG_FILE_SEPARATOR
# import sensors.detectron

_C.TASK.DETECTRON_SENSOR = CN()
_C.TASK.DETECTRON_SENSOR.TYPE = 'detectron_sensor'
_C.TASK.DETECTRON_SENSOR.DEVICE = 'cuda'

_C.TASK.MULTI_SPL = CN()
_C.TASK.MULTI_SPL.TYPE = 'MULTI_SPL'
_C.TASK.MULTI_SPL.SUCCESS_DISTANCE = 0.2

def get_config( config_paths=None, opts=None):
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
