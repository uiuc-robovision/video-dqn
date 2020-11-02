import argparse
from disk_logger import DiskReader
from evaluation.policy_defaults import name_from_config, load_file

def display_results(config):
    log_folder = f'{config.RESULT_LOCATION}/{name_from_config(config)}'
    print('Log Folder:', log_folder)
    logger = DiskReader(log_folder)
    dat = logger.data()
    for k in sorted(dat.keys()):
        print(f'Episode {k}: SPL {dat[k]}')
    mean = sum(dat.values())/len(dat)
    print(f'Mean SPL: {mean}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='simulate policy')
    parser.add_argument('config', help='folder containing config file')
    args = parser.parse_args()
    config = load_file(args.config)
    display_results(config)
