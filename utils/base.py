import os
import os.path as osp
import torch
import time
import json
import yaml
import shutil
import random
import numpy as np

class ConfigParser():
    def __init__(self, args):
        """
        class to parse configuration.
        """
        args = args.parse_args()
        self.cfg = self.merge_config_file(args)
        for k, v in args.__dict__.items():
            if isinstance(v, str) and '[' in v and ']' in v:
                args.__dict__[k] = str_to_list(v)
            if isinstance(v, str) and (v.lower() == 'null' or v.lower() == 'none'):
                args.__dict__[k] = None
            if isinstance(v, str) and is_number(v):
                args.__dict__[k] = float(v)

        # set random seed
        self.set_seed()

    def __str__(self):
        return str(self.cfg.__dict__)

    def __getattr__(self, name):
        """
        Access items use dot.notation.
        """
        return self.cfg.__dict__[name]

    def __getitem__(self, name):
        """
        Access items like ordinary dict.
        """
        return self.cfg.__dict__[name]

    def merge_config_file(self, args, allow_invalid=True):
        """
        Load json config file and merge the arguments
        """
        assert args.config is not None
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            if 'config' in cfg.keys():
                del cfg['config']
        f.close()
        invalid_args = list(set(cfg.keys()) - set(dir(args)))
        if invalid_args and not allow_invalid:
            raise ValueError(f"Invalid args {invalid_args} in {args.config}.")
        
        for k in list(cfg.keys()):
            if k in args.__dict__.keys() and args.__dict__[k] is not None:
                print('=========>  overwrite config: {} = {}'.format(k, args.__dict__[k]))
                del cfg[k]
        args.__dict__.update(cfg)

        return args

    def set_seed(self):
        ''' set random seed for random, numpy and torch. '''
        if 'seed' not in self.cfg.__dict__.keys():
            return
        if self.cfg.seed is None:
            self.cfg.seed = int(time.time()) % 1000000
        print('=========>  set random seed: {}'.format(self.cfg.seed))
        # fix random seeds for reproducibility
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)

    def save_codes_and_config(self, save_path):
        """
        save codes and config to $save_path.
        """
        cur_codes_path = osp.dirname(osp.dirname(os.path.abspath(__file__)))
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        shutil.copytree(cur_codes_path, osp.join(save_path, 'codes'), \
            ignore=shutil.ignore_patterns('*debug*', '*data*', '*outputs*', '*exps*', '*.txt', '*.json', '*.mp4', '*.png', '*.jpg', '*.ply', '*.csv', '*.pth', '*.tar', '*.npz'))

        with open(osp.join(save_path, 'config.yaml'), 'w') as f:
            f.write(yaml.dump(self.cfg.__dict__))
        f.close()


### common utils
def list_to_str(l, dot='_'):
    if isinstance(l, (list, tuple, np.ndarray)):
        return dot.join(str(i) for i in l)
    elif isinstance(l, int):
        return str(l)
    elif isinstance(l, str):
        return list_to_str(str_to_list(l))
    else:
        raise ValueError()

def str_to_list(l):
    return json.loads(l)

def is_pow2(x):
    return x > 0 and (x & (x - 1)) == 0

def is_pow3(x):
    return x > 0 and abs(np.power(x, 1/3) - np.round(np.power(x, 1/3))) < 1e-5

def is_number(s):
    try: 
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

### query utils
def batchify_query(query_fn, inputs: torch.Tensor, chunk_size: int, dim_batchify: int=0):
    raw_out = []
    for i in range(0, inputs.shape[0], chunk_size):
        out_i = query_fn(inputs[i:i+chunk_size])
        if not isinstance(out_i, tuple):
            out_i = [out_i]
        raw_out.append(out_i)

    ret = []
    for out in zip(*raw_out):
        out = torch.cat(out, dim=dim_batchify)
        ret.append(out)

    return ret[0] if len(ret) == 1 else tuple(ret)

def batchify_query_np(query_fn, inputs: torch.Tensor, chunk_size: int, dim_batchify: int=0):
    raw_out = []
    # for i in tqdm(range(0, inputs.shape[0], chunk_size)):
    for i in range(0, inputs.shape[0], chunk_size):
        out_i = query_fn(inputs[i:i+chunk_size])
        if not isinstance(out_i, tuple):
            out_i = [out_i]
        tmp = []
        for j in range(len(out_i)):
            tmp += [out_i[j].detach().cpu().numpy()]
        raw_out.append(tmp)

    ret = []
    for out in zip(*raw_out):
        out = np.concatenate(out, axis=dim_batchify)
        ret.append(out)

    return ret[0] if len(ret) == 1 else tuple(ret)


### other utils
def sample_neighbors(offsets, allow_center=True, allow_diagonals=True):
    ret = []
    if isinstance(offsets, list):
        offsets = np.array(offsets)
    for i_x, x in enumerate(offsets):
        for i_y, y in enumerate(offsets):
            for i_z, z in enumerate(offsets):
                if not allow_center and x == 0 and y == 0 and z == 0:
                    continue
                if not allow_diagonals and (i_x == 0 or i_x == offsets.shape[0]-1) and (i_y == 0 or i_y == offsets.shape[0]-1)and (i_z == 0 or i_z == offsets.shape[0]-1):
                    continue
                ret.append(torch.Tensor([[x, y, z]]))
    return torch.cat(ret, dim=0)

