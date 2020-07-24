import os
import yaml
import torch
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import re
from defaults import get_cfg_defaults

# valid_values = {'LOSS_CLIP': ['sigmoid','rect','none'], 'DATASET': ['nobias','bias_single','reacher','reacher_replay','reacher_replay_override','gibson','gibson_medium','gibson_medium_40k','gibson_medium_noisy_40k','gibson_medium_noisy', 'gibson_medium_noisy_value', 'gibson_medium_noisy_40k_value','gibson_branch','real','multi_target','multi_target_longer']}
valid_values = {'LOSS_CLIP': ['sigmoid','rect','none']}

class ExperimentConfig():
    def __init__(self, folder, device=None,remove=False,resume=False, run_prefix='run',tensorboard=True):
        super(ExperimentConfig, self).__init__()
        self.folder = folder
        if remove:
            os.system(f'rm -r {folder}/{run_prefix}*')

        self.files = os.popen(f'ls {folder}').read().split('\n')
        max_run = 0
        for f in self.files:
            match = re.search(f'^{run_prefix}(\d+)$',f)
            if match:
                max_run = int(match[1])

        if not resume:
            max_run += 1

        log_dir = f'{folder}/{run_prefix}{max_run}'

        if tensorboard:
            self.writer = SummaryWriter(log_dir =log_dir, comment = log_dir)
        self.cfg = get_cfg_defaults()
        self.cfg.merge_from_file(f'{folder}/config.yml')
        self.cfg.freeze()

        for k in valid_values:
            if self.cfg[k] not in valid_values[k]:
                raise Exception(f"Invalid value for {k}")

        for k in self.cfg:
            setattr(self,k,self.cfg[k])

        self.gpu_id = None
        if device is not None:
            self.device = torch.device(device)
            matches = re.match('.*:(\d+)',device)
            if matches:
                self.gpu_id = int(matches[1])
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    model = ExperimentConfig('logs/test', ['log_space', 'panorama'])
