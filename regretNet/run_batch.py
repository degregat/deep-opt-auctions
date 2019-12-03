#!/usr/bin/env python3

from pathlib import Path
from itertools import product
from functools import partial
from multiprocessing import Pool
import argparse
from ruamel.yaml import YAML
import copy
import signal

import tensorflow as tf

import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

import git

from nets import *
from cfgs import *
from data import *
from clip_ops.clip_ops import *
from trainer import *

# DEBUG
#import pudb; pu.db
#from tensorflow.python import debug as tf_debug


# Initialize yaml serializer
from easydict import EasyDict as edict
yaml = YAML()
yaml.register_class(edict)

# Initialize argument parser

# Supported settings
settings = [ "additive_1x2_uniform",
             "additive_5x10_uniform"]

# Arguments
parser = argparse.ArgumentParser(prog='Run experiments of deep-opt-auctions with differential privacy')
parser.add_argument('--setting', type=str, nargs='?', choices=settings, required=True)
parser.add_argument('--noise-vals', type=float, nargs='+', help='Array of noise_multiplier values')
parser.add_argument('--clip-vals', type=float, nargs='+', help='Array of l2_norm_clip values')
parser.add_argument('--iterations', type=int, nargs='?', help='Number of iterations for each instance')
parser.add_argument('--mpc-off', type=bool, nargs='+', help='Toggle MPC')
parser.add_argument('--add-no-dp-run', action='store_true', help='Run one instance with no differential privacy')
parser.add_argument('--description', type=str, nargs='?', help='Short description of the batch, should be unique')
parser.add_argument('--parallel', type=int, nargs='?', help='How many trainers or tests to run in parallel')
args = parser.parse_args()

# Helper functions
def list_to_str(vals):
    return(' '.join(map(str, vals)))

# The following two functions are not part of Batch, since object
# methods don't pickle, which multiprocessing requires
def train(setting, cfg):
    try:
        tf.reset_default_graph()

        if setting == "additive_1x2_uniform":
            cfg = cfg
            Net = additive_net.Net
            Generator = uniform_01_generator.Generator
            clip_op_lambda = (lambda x: clip_op_01(x))
            Trainer = trainer.Trainer

        if setting == "additive_5x10_uniform":
            cfg = cfg
            Net = additive_net.Net
            Generator = uniform_01_generator.Generator
            clip_op_lambda = (lambda x: clip_op_01(x))
            Trainer = trainer.Trainer

        net = Net(cfg)
        generator = [Generator(cfg, 'train'), Generator(cfg, 'val')]
        m = trainer.Trainer(cfg, "train", net, clip_op_lambda)
        m.train(generator)

    # Ignore keyboard interrupts and pass them to the parent process
    # for correct abort
    except KeyboardInterrupt:
        pass

def test(setting, cfg):
    try:
        # Run the test for each save point to gather data about
        # convergence and DP properties
        for save in range(1000, cfg.train.max_iter+1, 1000):
            tf.reset_default_graph()

            if setting == "additive_1x2_uniform":
                cfg = cfg
                Net = additive_net.Net
                Generator = uniform_01_generator.Generator
                clip_op_lambda = (lambda x: clip_op_01(x))
                Trainer = trainer.Trainer

            if setting == "additive_5x10_uniform":
                cfg = cfg
                Net = additive_net.Net
                Generator = uniform_01_generator.Generator
                clip_op_lambda = (lambda x: clip_op_01(x))
                Trainer = trainer.Trainer

            cfg.test.restore_iter = save
            net = Net(cfg)
            generator = Generator(cfg, 'test')
            m = trainer.Trainer(cfg, "test", net, clip_op_lambda)
            m.test(generator)

    except KeyboardInterrupt:
        pass

class Batch():
    def __init__(self, args):
        self.setting = args.setting
        self.noise_vals = args.noise_vals
        self.clip_vals = args.clip_vals
        self.iterations = args.iterations
        self.add_no_dp_run = args.add_no_dp_run
        self.description = args.description
        self.mpc_off = args.mpc_off

        # How many trainers/tests to run in parallel
        self.parallel = args.parallel

        # Commit all models instead of only the last one
        self.commit_all = False

        # Run shorter tests
        self.test_fast = True

        self.configs = []

        if self.setting == "additive_1x2_uniform":
            self.base_cfg = additive_1x2_uniform_config.cfg
        if self.setting == "additive_5x10_uniform":
            self.base_cfg = additive_5x10_uniform_config.cfg

        if not Path('batch_experiments').exists():
            Path('batch_experiments').mkdir()

    def set_batch_num(self):
        batches = Path('batch_experiments').glob(self.setting+'*')
        batch_nums = [ int(str(b).partition('_batch_')[2]) for b in batches ]
        if batch_nums:
            return(max(batch_nums)+1)
        else:
            return(1)

    def set_batch_id(self):
        return(self.setting + '_batch_' + str(self.batch_num))

    def set_batch_dir(self):
        return(Path('batch_experiments') / self.batch_id)

    def make_dirs(self):
        self.batch_dir.mkdir()
        (self.batch_dir/'cfgs').mkdir()

    def write_cmd(self):
        with open(str(self.batch_dir/ (self.batch_id + '_cmd.sh')), 'w') as cmdfile:
            cmdfile.write('run_batch.py --setting ' + self.setting +
                          (' --noise-vals ' + list_to_str(self.noise_vals) if self.noise_vals else '' ) +
                          (' --clip-vals ' + list_to_str(self.clip_vals) if self.clip_vals else '' ) +
                          (' --iterations ' + str(self.iterations) if self.iterations else '' ) +
                          (' --add-no-dp-run' if self.add_no_dp_run else '') +
                          (' --description ' + self.description if self.description else '') + '\n')

    def get_run_id(self, noise, clip):
        return(self.setting + '_noise_' + str(noise) + '_clip_' + str(clip) + '_iter_' + str(self.iterations))

    def get_cfg_path(self, noise, clip):
        return(Path('cfgs') / (self.get_run_id(noise, clip) + '.cfg'))

    def check_flags_and_set_to_iter(self, flags):
        for flag in flags:
            if flag > self.iterations:
                flag = self.iterations

    def gen_config(self, noise, clip):
        cfg = self.base_cfg

        cfg.dir_name = str(self.batch_dir / self.get_run_id(noise, clip))

        cfg.train.max_iter = self.iterations
        # Parameters for differentially private optimizer
        cfg.train.noise_multiplier = noise
        cfg.train.l2_norm_clip = clip

        # Save the model often to get more datapoints from tests
        cfg.train.save_iter = 1000
        cfg.train.max_to_keep = int(self.iterations / 1000)

        # MPC
        if self.mpc_off:
            cfg.mpc = False
        
        # Test flags
        if self.test_fast :
            cfg.test.num_misreports = 1
            cfg.test.gd_iter = 0
            #cfg.test.num_batches
            #cfg.test.batch_size

        # Dependent flags
        dependent_flags = [
            cfg.val.print_iter,
            cfg.test.restore_iter
        ]
        self.check_flags_and_set_to_iter(dependent_flags)

        # Serialize and save each config for reproducibility. This
        # does not preserve the dictionary order in python 2.7
        yaml.dump(cfg, self.batch_dir / self.get_cfg_path(noise, clip))
        return(copy.deepcopy(cfg))

    # Generate configs for each parameter combination
    def gen_configs(self):
        if self.add_no_dp_run:
            self.configs.append(self.gen_config(None, None))
        if self.noise_vals and self.clip_vals:
            for noise, clip in product(self.noise_vals, self.clip_vals):
                self.configs.append(self.gen_config(noise, clip))

    # Run the trainer once for each config that was generated
    def train_all(self):
        pool = Pool(self.parallel)

        try:
            pool.map(partial(train, self.setting), self.configs)

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            pass

    # Run the test once for each model that was generated
    def test_all(self):
        pool = Pool(self.parallel)

        try:
            pool.map(partial(test, self.setting), self.configs)

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            pass

    def get_exp_dirs(self):
        exp_dirs = [ self.batch_dir / self.get_run_id(noise, clip)
                     for (noise, clip) in product(self.noise_vals, self.clip_vals) ]
        if self.add_no_dp_run:
            exp_dirs = exp_dirs + [ self.batch_dir / self.get_run_id(None, None) ]
        return(exp_dirs)

    def accumulate_train_data(self):
        train_data = pd.concat([ pd.read_csv(Path(exp_dir) / 'train_data.csv' )
                                 for exp_dir in self.exp_dirs ],
                               ignore_index=True)
        train_data.to_csv(self.batch_dir / 'train_data.csv')

    def accumulate_test_data(self):
        test_data = pd.concat([ pd.read_csv(Path(exp_dir) / Path('iter_' + str(i) + '_test_data.csv'))
                                for exp_dir in self.exp_dirs
                                for i in range(1000, self.iterations+1, 1000) ],
                              ignore_index = True)
        test_data.to_csv(self.batch_dir / 'test_data.csv')

    def visualize(self):
        with open('visualize_batch.ipynb') as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': str(self.batch_dir) }})

        with open(str(self.batch_dir / Path('visualize_' + self.batch_id + '.ipynb')), 'w') as f:
            nbformat.write(nb, f)

    def commit_code(self):
        self.git = git.Git(Path.cwd().parent)
        self.repo = git.Repo(Path.cwd().parent)
        self.msg = (self.description + '\n' if self.description else '') + 'Setting: ' + self.setting + '\nNoise: ' + list_to_str(self.noise_vals) + '\nClip: ' + list_to_str(self.clip_vals) + '\nIterations: ' + str(self.iterations)

        # Commit code changes
        self.former_branch = self.repo.active_branch

        if self.repo.is_dirty():
            self.git.commit('-a', '-m', self.msg)
            self.code_changed = True
        else:
            self.code_changed = False

        if self.description:
            self.result_branch = 'exp_' + self.description
        else:
            self.result_branch = self.batch_id

        try:
            self.git.checkout(self.result_branch)
        except git.exc.GitError:
            self.git.checkout('-b', self.result_branch)

        self.git.pull('.', self.former_branch)


    def commit_batch(self):
        # Track data, command and visualization
        out = [ self.batch_dir / 'test_data.csv',
                self.batch_dir / 'train_data.csv',
                self.batch_dir / (self.batch_id + '_cmd.sh'),
                self.batch_dir / ('visualize_' + self.batch_id + '.ipynb') ]
        # Only track the last version of every model
        models = [ model for exp_dir in self.exp_dirs for model in exp_dir.glob('model-'+str(self.iterations)+'*') ]

        if self.commit_all == False:
            # FIXME: Only use GitPython as git.add fails with some git versions
            self.git.add(*(out))
            self.git.add(*(models))
        elif self.commit_all == True:
            git.add(self.batch_dir)

        # Commit experiment data
        if self.code_changed == True:
            self.git.commit('--amend', '--no-edit')
        else:
            self.git.commit('-a', '-m', self.msg)

        # Return to former branch
        self.git.checkout(self.former_branch)

    def run_batch(self):
        try:
            self.batch_num = self.set_batch_num()
            self.batch_id = self.set_batch_id()
            self.batch_dir = self.set_batch_dir()

            #self.commit_code()

            self.make_dirs()
            self.write_cmd()
            self.gen_configs()

            self.train_all()
            self.test_all()
            self.exp_dirs = self.get_exp_dirs()
            self.accumulate_train_data()
            self.accumulate_test_data()
            self.visualize()
            #self.commit_batch()

        except Exception as e:
            #self.git.checkout(self.former_branch)
            print(e)

        except KeyboardInterrupt as e:
            #self.git.checkout(self.former_branch)
            print(e)

        except Warning:
            pass

batch = Batch(args)
batch.run_batch()
