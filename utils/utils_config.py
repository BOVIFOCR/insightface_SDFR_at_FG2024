import importlib
import os.path as osp
from datetime import datetime


# def get_config(config_file):
def get_config(config_file, run_name=''):
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = osp.basename(config_file)
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module("configs.base")
    cfg = config.config
    config = importlib.import_module("configs.%s" % temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    if cfg.output is None:
        curr_date_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # cfg.output = osp.join('work_dirs', temp_module_name)
        cfg.output = osp.join('work_dirs', temp_module_name, curr_date_time, run_name)
    return cfg