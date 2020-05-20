#!/usr/bin/env python2

from pathlib import Path
from itertools import product
from functools import partial
from multiprocessing import Pool
from multiprocessing import Process
import argparse
from ruamel.yaml import YAML
import copy
import signal
import shutil

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

# Initialize yaml serializer
from easydict import EasyDict as edict
yaml = YAML()
yaml.register_class(edict)

# Initialize argument parser

# Supported settings
settings = [ "additive_1x2_uniform",
             "additive_5x10_uniform",
             "additive_5x10_reports",
             "additive_5x10_misreports",
             "additive_5x10"]

# Arguments
parser = argparse.ArgumentParser(prog='Run experiments of deep-opt-auctions with differential privacy')
parser.add_argument('--setting', type=str, nargs='?', choices=settings, required=True)
parser.add_argument('--noise-vals', type=float, nargs='+', help='Array of noise_multiplier values')
parser.add_argument('--clip-vals', type=float, nargs='+', help='Array of l2_norm_clip values')
parser.add_argument('--iterations', type=int, nargs='?', help='Number of iterations for each instance')
parser.add_argument('--add-no-dp-run', action='store_true', help='Run one instance with no differential privacy')
parser.add_argument('--description', type=str, nargs='?', help='Short description of the batch, should be unique')
parser.add_argument('--reports', type=str, nargs='?', help='Saved reports of another run')
args = parser.parse_args()

# Helper functions
def list_to_str(vals):
    return(' '.join(map(str, vals)))


# The following two functions are not part of Batch, since object
# methods don't pickle, which multiprocessing requires
# Always keep cfg as the last argument since they are what are mapping over
def train(setting, reports, cfg):
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

        if setting == "additive_5x10_reports":
            cfg = cfg
            Net = additive_net.Net
            Generator = nonmisreports.Generator
            clip_op_lambda = (lambda x: clip_op_01(x))
            Trainer = trainer.Trainer

        if setting == "additive_5x10_misreports":
            cfg = cfg
            Net = additive_net.Net
            Generator = misreports.Generator
            clip_op_lambda = (lambda x: clip_op_01(x))
            Trainer = trainer.Trainer

        net = Net(cfg)
        generator = [Generator(cfg, 'train', reports), Generator(cfg, 'train', reports)] # TODO: second generator was in 'val' mode
        m = trainer.Trainer(cfg, "train", net, clip_op_lambda)
        m.train(generator)

    # Ignore keyboard interrupts and pass them to the parent process
    # for correct abort
    except KeyboardInterrupt:
        pass

def test(setting, reports, cfg):
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

            if setting == "additive_5x10_reports":
                cfg = cfg
                Net = additive_net.Net
                Generator = nonmisreports.Generator
                clip_op_lambda = (lambda x: clip_op_01(x))
                Trainer = trainer.Trainer

            if setting == "additive_5x10_misreports":
                cfg = cfg
                Net = additive_net.Net
                Generator = misreports.Generator
                clip_op_lambda = (lambda x: clip_op_01(x))
                Trainer = trainer.Trainer

            cfg.test.restore_iter = save
            net = Net(cfg)
            generator = Generator(cfg, 'train', reports) # TODO: this was 'test'
            m = trainer.Trainer(cfg, "test", net, clip_op_lambda)
            m.test(generator)

    except KeyboardInterrupt:
        pass

# TODO: run all this in one parent directory
# TODO: visualize in one notebook, overlaying truthful and misreporting runs
# TODO: either fix clip/noise or facet agent utilities over both

# Any distribution with full support should work for misreport calculation. Uniform is robust.
# TODO: make opt_1/opt_2 DP
# TODO: tune noise/clipping parameters
# TODO: tune learning rate

# TODO: check utilities from vanilla code
# TODO: check wether 1x2 case equals 2nd price auction

class Experiment():
    def __init__(self, args):
        self.args = args
        if args.reports:
            self.reports = np.load(args.reports)
        else:
            self.reports = self.gen_reports(5,10)

        if not Path('batch_experiments').exists():
            Path('batch_experiments').mkdir()

    # Generate initial true valuation reports
    def gen_reports(self, agents, items):
        reports = np.random.rand(agents, items)
        return reports    

    def set_exp_num(self):
        exps = Path('batch_experiments').glob(self.args.setting+'*')
        exp_nums = [ int(str(e).partition('_exp_')[2]) for e in exps ]
        if exp_nums:
            return(max(exp_nums)+1)
        else:
            return(1)
        return

    def set_exp_id(self):
        return(self.args.setting + '_exp_' + str(self.exp_num))
        
    def set_exp_dir(self):
        return(Path('batch_experiments') / self.exp_id)

    def save_reports(self):
        np.save((Path(self.exp_dir) / 'reports.npy'), self.reports)
    
    def aggregate_data(self):
        files = ['utility_data.csv', 'train_data.csv', 'test_data.csv']
        for f in files:
            shutil.copy(str(self.truthful_batch.get_batch_dir() / f) ,
                        str(self.exp_dir / Path('truthful_' + f)))
            shutil.copy(str(self.misreport_batch.get_batch_dir() / f),
                        str(self.exp_dir / Path('misreport_' + f)))

    def visualize(self):
        return
    
    def run_exp(self):
        try:
            self.exp_num = self.set_exp_num()
            self.exp_id = self.set_exp_id()
            self.exp_dir = self.set_exp_dir()
            self.exp_dir.mkdir()
            self.save_reports()

            self.truthful_batch = Batch(self.args, 1, self.reports, self.exp_dir)
            self.misreport_batch = Batch(self.args, 0, self.reports, self.exp_dir)

            t_run = Process(target=self.truthful_batch.run_batch())
            m_run = Process(target=self.misreport_batch.run_batch())
            t_run.start()
            m_run.start()
            t_run.join()
            m_run.join()

            self.aggregate_data()

        except Exception as e:
            print(e)
            
        except KeyboardInterrupt as e:
            print(e)

        except Warning:
            pass
        

    
class Batch():
    def __init__(self, args, truthful, reports, exp_dir):
        if args.setting == "additive_5x10":
            if truthful:
                self.setting = "additive_5x10_reports"
            elif not truthful:
                self.setting = "additive_5x10_misreports"

        self.noise_vals = args.noise_vals
        self.clip_vals = args.clip_vals
        self.iterations = args.iterations
        self.add_no_dp_run = args.add_no_dp_run
        self.description = args.description

        self.truthful = truthful
        self.reports = reports

        self.exp_dir = exp_dir
        
        # How many trainers/tests to run in parallel
        self.parallel_trainers = 4
        self.parallel_tests = 4

        # Commit all models instead of only the last one
        self.commit_all = False

        # Run shorter tests
        self.test_fast = True

        # TODO: Batch sizes need to be the same. Check why it breaks.
        self.batch_size = 1
        self.test_batch_size = 1

        self.configs = []

        if self.setting == "additive_1x2_uniform":
            self.base_cfg = additive_1x2_uniform_config.cfg
        if self.setting == "additive_5x10_uniform":
            self.base_cfg = additive_5x10_uniform_config.cfg
        if self.setting == "additive_5x10_reports":
            self.base_cfg = additive_5x10_uniform_config.cfg
        if self.setting == "additive_5x10_misreports":
            self.base_cfg = additive_5x10_uniform_config.cfg


    def set_batch_num(self):
        batches = Path('batch_experiments') / self.exp_dir.glob(self.setting+'*')
        batch_nums = [ int(str(b).partition('_batch_')[2]) for b in batches ]
        if batch_nums:
            return(max(batch_nums)+1)
        else:
            return(1)

    def set_batch_id(self):
        return(self.setting)

    def set_batch_dir(self):
        return(self.exp_dir / self.batch_id)

    def get_batch_dir(self):
        return(self.batch_dir)

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
        cfg.train.max_to_keep = self.iterations / 1000

        cfg.train.batch_size = self.batch_size
        cfg.test.batch_size = self.test_batch_size

        # Test flags
        if self.test_fast :
            cfg.test.num_misreports = 1
            cfg.test.gd_iter = 0
            #cfg.test.num_batches
            #cfg.test.batch_size

        # To save alloc and pay from test
        #cfg.train.data = "online"
        #cfg.train.adv_reuse = False
        #cfg.test.data = "fixed"
        cfg.test.save_output = True

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

    def save_reports(self):
        np.save(self.batch_dir / Path('reports.npy'), self.reports)

    def stack_reports(self, batch_size):
        return np.stack((self.reports,)* batch_size)

    # Run the trainer once for each config that was generated
    def train_all(self):
        pool = Pool(self.parallel_trainers)

        try:
            pool.map(partial(train, self.setting, self.stack_reports(self.batch_size)), self.configs)

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            pass

    # Run the test once for each model that was generated
    def test_all(self):
        pool = Pool(self.parallel_trainers)

        try:
            pool.map(partial(test, self.setting, self.stack_reports(self.test_batch_size)), self.configs)

        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            pass

    # TODO: Fix noise, clip passing by calculating utility in the trainer
    def get_run_dirs(self):
        run_dirs = [ (self.batch_dir / self.get_run_id(noise, clip), noise, clip)
                     for (noise, clip) in product(self.noise_vals, self.clip_vals) ]
        if self.add_no_dp_run:
            run_dirs = run_dirs + [ (self.batch_dir / self.get_run_id(None, None), 0, 0) ]
        return(run_dirs)

    def accumulate_train_data(self):
        train_data = pd.concat([ pd.read_csv(Path(run_dir) / 'train_data.csv' )
                                 for run_dir, _, _ in self.run_dirs ],
                               ignore_index=True)
        train_data.to_csv(self.batch_dir / 'train_data.csv')

    def accumulate_test_data(self):
        test_data = pd.concat([ pd.read_csv(Path(run_dir) / Path('iter_' + str(i) + '_test_data.csv'))
                                for run_dir, _, _ in self.run_dirs
                                for i in range(1000, self.iterations+1, 1000) ],
                              ignore_index = True)
        test_data.to_csv(self.batch_dir / 'test_data.csv')

    def compute_utility(self):
        columns = ["Iter", "Noise", "Clip", "Agent", "Utility"]
        utility_array=[]
        for run_dir in self.get_run_dirs():
            dir, noise, clip = run_dir
            reports = list(np.load(self.batch_dir / Path('reports.npy')))

            for i in range(1000, self.iterations+1, 1000):
                pay = np.load(Path(dir) / Path('pay_tst_' + str(i) + '.npy'))
                alloc = np.load(Path(dir) / Path('alloc_tst_' + str(i) + '.npy'))
                utilities = np.sum(alloc[0] * reports, axis=-1) - pay[0] #When batch size is 1, remove array access
                print(utilities)
                for agent, utility in enumerate(utilities):
                    utility_array.append([i, noise, clip, agent, utility])

        pd.DataFrame(utility_array, columns=columns).to_csv(Path(self.batch_dir / Path('utility_data.csv')))

    def visualize(self):
        with open('visualize_batch.ipynb') as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python2')
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
            self.result_branch = 'run_' + self.description
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
        models = [ model for run_dir in self.run_dirs for model in run_dir.glob('model-'+str(self.iterations)+'*') ]

        if self.commit_all == False:
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
            #self.batch_num = self.set_batch_num()
            self.batch_id = self.set_batch_id()
            self.batch_dir = self.set_batch_dir()

            #self.commit_code()

            self.make_dirs()

            self.write_cmd()
            self.gen_configs()

            self.save_reports()
            self.train_all()
            self.test_all()
            self.run_dirs = self.get_run_dirs()
            self.accumulate_train_data()
            self.accumulate_test_data()
            self.compute_utility()
            self.visualize()
            ##self.commit_batch()

        except Exception: # as e:
            #self.git.checkout(self.former_branch)
            #print e
            pass

        except KeyboardInterrupt: # as e:
            #self.git.checkout(self.former_branch)
            #print e
            pass

        except Warning:
            pass

    def batch_dir(self):
        return(self.batch_dir)

exp = Experiment(args)
exp.run_exp()
