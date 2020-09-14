#!/usr/bin/env python3

import pdb

from pathlib import Path
from itertools import product
from functools import partial
from multiprocessing.pool import Pool
from multiprocessing import Process
import argparse
#from ruamel.yaml import YAML
import copy
import signal
import shutil
import time

import tensorflow as tf

import pandas as pd
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

#import git

from nets import *
from cfgs import *
from data import *
from clip_ops.clip_ops import *
from trainer import *

# Initialize argument parser

# Supported settings
# TODO: cleanup
settings = [ "additive_5x10_reports",
             "additive_5x10_misreports",
             "additive_5x10"]

# Arguments
parser = argparse.ArgumentParser(prog='Run experiments of deep-opt-auctions with differential privacy')
parser.add_argument('--setting', type=str, nargs='?', choices=settings, required=True)
parser.add_argument('--noise-vals', type=float, nargs='+', help='Array of noise_multiplier values')
parser.add_argument('--clip-vals', type=float, nargs='+', help='Array of l2_norm_clip values')
parser.add_argument('--iterations', type=int, nargs='?', default=1000, help='Number of iterations for each instance')
parser.add_argument('--interval', type=int, nargs='?', default=50, help='iter interval between model saves/restores. Determines granularity of observations.')
parser.add_argument('--add-no-dp-run', action='store_true', help='Run one instance with no differential privacy')
parser.add_argument('--description', type=str, nargs='?', help='Short description of the batch, should be unique')
parser.add_argument('--valuations', type=str, nargs='?', help='Saved valuations of another run')
parser.add_argument('--pool-size', type=int, nargs='?', default=4, help='Number of parallel processes')
args = parser.parse_args()

# Helper functions
def list_to_str(vals):
    return(' '.join(map(str, vals)))

# The following two functions are not part of Batch, since object
# methods don't pickle, which multiprocessing requires
# Always keep cfg as the last argument since they are what are mapping over

# TODO: refactor fixed/online data generation
def train(setting, cfg_and_reports):
    try:
        cfg, reports_once = cfg_and_reports
        reports = np.stack((reports_once,)* cfg.train.batch_size)
        tf.compat.v1.reset_default_graph()

        if setting == "additive_5x10":
            cfg = cfg
            Net = additive_net.Net
            Generator = fixed_reports.Generator
            clip_op_lambda = (lambda x: clip_op_01(x))
            Trainer = trainer.Trainer
        
        net = Net(cfg)
        generator = [Generator(cfg, 'train', reports), Generator(cfg, 'train', reports)] # NOTE: second generator was in 'val' mode
        m = trainer.Trainer(cfg, "train", net, clip_op_lambda)
        m.train(generator)

    # Ignore keyboard interrupts and pass them to the parent process
    # for correct abort
    except KeyboardInterrupt:
        pass

def test(setting, cfg_and_reports):
    try:
        # Run the test for each save point to gather data about
        # convergence and DP properties
        cfg, reports_once = cfg_and_reports
        reports = np.stack((reports_once,)* cfg.train.batch_size)

        for save in range(args.interval, cfg.train.max_iter+1, args.interval):
            tf.compat.v1.reset_default_graph()

            if setting == "additive_5x10":
                cfg = cfg
                Net = additive_net.Net
                Generator = fixed_reports.Generator
                clip_op_lambda = (lambda x: clip_op_01(x))
                Trainer = trainer.Trainer
                
            cfg.test.restore_iter = save
            net = Net(cfg)
            generator = Generator(cfg, 'train', reports) # NOTE: this was 'test'
            m = trainer.Trainer(cfg, "test", net, clip_op_lambda)
            m.test(generator)
            
    except KeyboardInterrupt:
        pass

    except Exception as err:
        print(err)
    

# Any distribution with full support should work for misreport calculation. Uniform is robust.

class Experiment():
    def __init__(self, args):
        self.args = args

        self.agents = 5
        self.items = 10
        
        # Generate true Valuations
        if args.valuations:
            self.valuations = np.load(args.valuations)
        else:
            self.valuations = self.gen_valuations()

        if not Path('batch_experiments').exists():
            Path('batch_experiments').mkdir()

    # Generate initial true valuation reports
    def gen_valuations(self):
        valuations = np.random.binomial(1, 0.5, (self.agents, self.items))
        return valuations

    def enum_misreports(self):
        # Enumerate all misreports (10x1 vectors with elements in
        # {0,1}). These are 1024.

        # Embed every vector in an array, so concatenate get arguments
        # with equal dimension.
        return ([ [[a, b, c, d, e, f, g, h, i, j]] for a in [0,1]
                 for b in [0,1]
                 for c in [0,1]
                 for d in [0,1]
                 for e in [0,1]
                 for f in [0,1]
                 for g in [0,1]
                 for h in [0,1]
                 for i in [0,1]
                 for j in [0,1]])

    # Fix all bids, but those of agent 0
    def gen_misreports(self, misreports):
        # Embeds misreports of agent 0 with fixed reports of the other agents.
        return np.concatenate((misreports, self.valuations[1:]))
        
    def set_exp_num(self):
        prefix = self.args.setting+"_exp_"
        exps = list(Path('batch_experiments').glob(prefix+'*'))
        if len(exps) == 0:
            return(1)
        else:
            exp_nums = [ int(str(e).partition('_exp_')[2]) for e in exps ]
            return(max(exp_nums)+1)

    def set_exp_id(self):
        return(self.args.setting + '_exp_' + str(self.exp_num))
        
    def set_exp_dir(self):
        return(Path('batch_experiments') / self.exp_id)

    def gen_batch_id(self, noise, clip):
        return(self.args.setting + '_noise_' + str(noise) + '_clip_' + str(clip))
    
    def save_valuations(self):
        np.save((Path(self.exp_dir) / 'valuations.npy'), self.valuations)

    def aggregate_data(self, dp_nc):        
        paths = []
        for noise, clip in dp_nc:
            batch_id = self.gen_batch_id(noise, clip)
            paths.append(self.exp_dir / batch_id)
            
        util = []
        tests = []
        train = []
        welfare = []
        for p in paths:
            util.append(pd.read_csv(p / Path('utility_data.csv'), index_col=None, header=0))
            tests.append(pd.read_csv(p / Path('test_data.csv'), index_col=None, header=0))
            train.append(pd.read_csv(p / Path('train_data.csv'), index_col=None, header=0))
            welfare.append(pd.read_csv(p / Path('welfare_data.csv'), index_col=None, header=0))
        pd.concat(util, axis=0, ignore_index=True).to_csv(Path(self.exp_dir) / 'utility.csv')
        pd.concat(tests, axis=0, ignore_index=True).to_csv(Path(self.exp_dir) / 'tests.csv')
        pd.concat(train, axis=0, ignore_index=True).to_csv(Path(self.exp_dir) / 'train.csv')
        pd.concat(welfare, axis=0, ignore_index=True).to_csv(Path(self.exp_dir) / 'welfare.csv')

    # TODO: Calculate f(eps) in the notebook that gets called here
    def visualize(self):
        with open('visualize_one_shot_misreports.ipynb') as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': str(self.exp_dir) }})

        with open(str(self.exp_dir /
                      Path('visualize_one_shot_misreports_exp_' + str(self.exp_num) + '.ipynb')), 'w') as f:
            nbformat.write(nb, f)

    def write_cmd(self):
        with open(str(self.exp_dir/ (self.exp_id + '_cmd.sh')), 'w') as cmdfile:
            cmd = ('run_batch.py --setting ' + args.setting +
                   ' --noise-vals ' + list_to_str(args.noise_vals) + #if args.noise_vals else '' ) +
                   ' --clip-vals ' + list_to_str(args.clip_vals) + # if args.clip_vals else '' ) +
                   ' --iterations ' + str(args.iterations) +
                   ' --interval ' + str(args.interval) +
                   ' --pool-size ' + str(args.pool_size) + 
                   (' --add-no-dp-run' if args.add_no_dp_run else '') + '\n')
            
            cmdfile.write(cmd)

    def run_exp(self):
        try:
            self.exp_num = self.set_exp_num()
            #self.exp_num =

            self.exp_id = self.set_exp_id()
            self.exp_dir = self.set_exp_dir()
            
            self.exp_dir.mkdir()
            self.write_cmd()
            self.save_valuations()
            
            # Use a new pool per batch to free memory. Same for misreports
            # Batch 0 is truthful, all subsequent ones are misreporting

            print("INIT")
            # Generate list of DP parameter tuples
            # TODO: Make dp_nc a field of Experiment
            dp_nc = list(product(self.args.noise_vals, self.args.clip_vals))
            if self.args.add_no_dp_run:
                dp_nc.append((None, None))

            print("GENERATE MISREPORTS")
            reports = []
            reports.append(self.valuations)

            # Enumerate misreports for agent0
            misreports = self.enum_misreports()#[:1] #uncomment for quick testing

            # Concatenate misreport for agent0 with valuations of
            # agents 1 through 4, for all misreports
            for misreport in misreports:
                reports.append(self.gen_misreports(misreport))
                
            #reports[0] now contains the valuations, reports[1]
            #through reports[1024] contain all possible misreports

            print("RUN BATCHES")
            for noise, clip in dp_nc:
                batch_id = self.gen_batch_id(noise, clip)
                Batch(self.args, reports, self.exp_dir, self.exp_num, noise, clip, batch_id).run_batch()

            print("AGGREGATE")
            self.aggregate_data(dp_nc)
            # TODO: Compute f(eps) deviation from IC, visualize it to quantify eps-IC
            self.visualize()

        except Exception as e:
            print(e)
            
        except KeyboardInterrupt as e:
            print(e)

        except Warning:
            pass

class Batch():
    # Each auction we learn is an overfit to a specific set of bids/reports.
    # For any agent, no bid should outperform any other bid by more
    # than f(eps), making the whole learner eps-IC.

    # In case of truthful auction, pass valuations as reports.
    # In non-truthful case, pass misreports.

    def __init__(self, args, reports, exp_dir, exp_num, noise, clip, batch_id):
        self.noise = noise
        self.clip = clip
        self.batch_id = batch_id

        self.setting = args.setting
        self.iterations = args.iterations
        self.add_no_dp_run = args.add_no_dp_run
        self.description = args.description
        self.interval = args.interval
        self.pool_size = args.pool_size
        
        #self.truthful = truthful
        self.reports = reports

        self.exp_dir = exp_dir
        self.exp_num = exp_num

        # How many trainers/tests to run in parallel
        self.parallel_trainers = self.pool_size
        self.parallel_tests = self.pool_size

        # Commit all models instead of only the last one
        self.commit_all = False

        # Run shorter tests
        self.test_fast = True

        # TODO: Batch sizes need to be the same, check why. (No issue at this time.)
        self.batch_size = 1
        self.test_batch_size = 1

        self.configs = []

        if self.setting == "additive_5x10":
            self.base_cfg = additive_5x10_uniform_config.cfg


    def set_batch_dir(self):
        return(self.exp_dir / self.batch_id)

    def get_batch_dir(self):
        return(copy.deepcopy(self.batch_dir))

    def make_dirs(self):
        self.batch_dir.mkdir()

    def get_run_id(self, report_num):
        # Run 0 are valuation reports
        # Run 1 to n are misreports
       
        if report_num == 0:
            return("valuation_report")
         
        if report_num != 0:
            return("misreport_" + str(report_num))

    def get_run_dir(self, report_num):
        return (self.batch_dir / self.get_run_id(report_num))
        
    def check_flags_and_set_to_iter(self, flags):
        for flag in flags:
            if flag > self.iterations:
                flag = self.iterations

    def gen_config(self, report_num):
        cfg = self.base_cfg

        cfg.dir_name = str(self.batch_dir / self.get_run_id(report_num))
        cfg.report_num = report_num
        cfg.exp_num = self.exp_num
        
        cfg.train.max_iter = self.iterations
        # Parameters for differentially private optimizer
        cfg.train.noise_multiplier = self.noise
        cfg.train.l2_norm_clip = self.clip

        # Write data and save the model often to get more datapoints from tests
        cfg.train.print_iter = self.interval
        cfg.train.save_iter = self.interval
        cfg.train.max_to_keep = int(self.iterations / self.interval)

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

        return(copy.deepcopy(cfg))

    # Generate configs for each parameter combination
    def gen_configs_and_reports(self):
        for report_num in range(0, len(self.reports)):
            self.configs.append(self.gen_config(report_num))

        # Transform to list, otherwise zipper gets consumed after
        # iterating over it
        self.configs_and_reports = list(zip(self.configs, self.reports))

    # Reports are deterministic
    def save_reports(self):
        np.save(self.batch_dir / Path('reports.npy'), self.reports)

    # Run the trainer once for each config that was generated
    def train_all(self):
        try:
            self.pool = Pool(processes=self.pool_size, maxtasksperchild=1)
            self.pool.imap(partial(train, self.setting), self.configs_and_reports)

            self.pool.close()
            self.pool.join()
            print("TRAINING DONE")
            
        except KeyboardInterrupt:
            self.pool.terminate()
            self.pool.join()
            pass

        except Exception as e:
            print(e)


    # Run the test once for each model that was generated
    def test_all(self):
        try:
            self.pool = Pool(processes=self.pool_size, maxtasksperchild=1)        
            self.pool.imap(partial(test, self.setting), self.configs_and_reports)
            self.pool.close()
            self.pool.join()
            
        except KeyboardInterrupt:
            self.pool.terminate()
            self.pool.join()
            pass

        except Exception as e:
            print(e)

    def get_run_dirs(self):
        run_dirs = [ (report_num, self.get_run_dir(report_num)) for report_num in range(0, len(self.reports)) ]

        self.run_dirs = run_dirs

    def accumulate_train_data(self):
        train_data = pd.concat([ pd.read_csv(Path(run_dir) / 'train_data.csv' )
                                 for _, run_dir in self.run_dirs ],
                               ignore_index=True)
        train_data.to_csv(self.batch_dir / 'train_data.csv')

    def accumulate_test_data(self):
        test_data = pd.concat([ pd.read_csv(Path(run_dir) / Path('iter_' + str(i) + '_test_data.csv'))
                                for _, run_dir in self.run_dirs
                                for i in range(self.interval, self.iterations+1, self.interval) ],
                              ignore_index = True)
        test_data.to_csv(self.batch_dir / 'test_data.csv')

    def compute_utility(self):
        columns = ["Exp", "Report", "Iter", "Noise", "Clip", "Agent", "Utility", "Regret"]
        utility_array=[]


        if (self.noise and self.clip) == None:
            noise, clip = 0, 0
        else:
            noise, clip = self.noise, self.clip
            
        for report_num, run_dir in self.run_dirs:
            valuations = list(np.load(self.exp_dir / Path('valuations.npy')))

            for i in range(self.interval, self.iterations+1, self.interval):
                pay = np.load(Path(run_dir) / Path('pay_tst_' + str(i) + '.npy'))
                alloc = np.load(Path(run_dir) / Path('alloc_tst_' + str(i) + '.npy'))
                utilities = np.sum(alloc[0] * valuations, axis=-1) - pay[0] #When batch size is 1, remove array access
                
                if report_num == 0:
                    self.val_utilities = utilities
                    for agent, utility in enumerate(utilities):
                        utility_array.append([self.exp_num, report_num, i, noise, clip, agent,
                                              utility, 0])

                else: 
                    regret = utilities - self.val_utilities
                    for agent, utility in enumerate(utilities):
                        utility_array.append([self.exp_num, report_num, i, noise, clip, agent,
                                              utility, regret[agent] ])
                
        pd.DataFrame(utility_array, columns=columns).to_csv(Path(self.batch_dir / Path('utility_data.csv')))
        

    def compute_welfare(self):
        # This computes only allocative welfare, not taking any transfers into account.
        columns = ["Exp", "Report", "Iter", "Noise", "Clip", "Welfare"]
        welfare_array=[]
        
        if (self.noise and self.clip) == None:
            noise, clip = 0, 0
        else:
            noise, clip = self.noise, self.clip
            
        for report_num, run_dir in self.run_dirs:
            valuations = list(np.load(self.exp_dir / Path('valuations.npy')))
            for i in range(self.interval, self.iterations+1, self.interval):
                alloc = np.load(Path(run_dir) / Path('alloc_tst_' + str(i) + '.npy'))
                welfare = np.sum(alloc[0] * valuations)
                welfare_array.append([self.exp_num, report_num, i, noise, clip, welfare])

        pd.DataFrame(welfare_array, columns=columns).to_csv(Path(self.batch_dir / Path('welfare_data.csv')))

    def run_batch(self):
        try:
            # batch_num only affects path and indexing

            print("INIT BATCH")
            self.batch_dir = self.set_batch_dir()
            
            self.make_dirs()

            print("GENERATE CFGS AND REPORTS")
            self.gen_configs_and_reports()

            print("TRAIN_ALL")
            self.train_all()

            print("TEST_ALL")
            self.test_all()

            self.get_run_dirs()
            print("ACCUMULATE TRAIN")
            self.accumulate_train_data()
            print("ACCUMULATE TEST")            
            self.accumulate_test_data()
            print("COMPUTE UTILITY")
            self.compute_utility()
            print("COMPUTE WELFARE")
            self.compute_welfare()

            
        except KeyboardInterrupt: # as e:
            #print e
            self.pool.close()
            self.pool.join()
            pass

        except Exception as e:
            print(e)
            pass

        except Warning:
            pass

    def batch_dir(self):
        return(self.batch_dir)

exp = Experiment(args)
exp.run_exp()
