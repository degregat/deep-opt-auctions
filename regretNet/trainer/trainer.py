import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from privacy.analysis import privacy_ledger
from privacy.analysis.rdp_accountant import compute_rdp_from_ledger
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer

import tf_encrypted as tfe

# Compatibility with tf 1 and 2 APIs for TensorFlow Privacy
try:
  GradientDescentOptimizer = tf.compat.v1.train.GradientDescentOptimizer
except:  # pylint: disable=bare-except
  GradientDescentOptimizer = tf.optimizers.SGD  # pylint: disable=invalid-name

# tf-encrypted config
# TODO: Add tfe configs for networking
#if len(sys.argv) > 1:
#  # config file was specified
#  config_file = sys.argv[1]
#  config = tfe.RemoteConfig.load(config_file)
#  tfe.set_config(config)
tfe.set_protocol(tfe.protocol.Pond()) 
#session_target = None

# Development notes
# Differentially private federated learning on MPC:
# Current state:
#    - Duplicate computation of inner loop to each agent
#    - securely aggregate (average) resulting lagrangian updates
#    - update and redistribute model
# Porting:
#    - Port to TF2.0, check which TODOs become obsolete by this
#    - Assess wether to use TFF for orchestration, logging, distributed generators/input
#    - Refactor logging (see LOGGING TODOs)
# Next steps for FL:
#    - Reduce local computation to simulate only one agent
#    - securely aggregate (sum) single agent lagrangian updates
#    - Analyze if and how to best federate inner loop

# TODO: Recover non-mpc codepath, including MPC and DP toggles in run_batch.py and configs
# TODO: Write (un)packing functions for grads and vars
# LOGGING TODO: For dev debugging log player updates and aggregated lagrangian

class Agent():
    def __init__(self, player_name, build_update_step, generator):
        self.player_name = player_name
        self._build_update_step = build_update_step
        self.train_gen, _ = generator
        # TODO: Split generator up and use one per agent for distributed computation

    @tfe.local_computation
    def compute_grads(self, iter, session):
        with tf.name_scope('gradient_computation'):
            # Drop variables to ease aggregation, we select scope again later
            # LOGGING TODO: log this
            print(self.player_name)

            # TODO: (un)packing helper functions
            player_lag_grads_and_vars = self._build_update_step(iter, session)
            
            grads, tf_vars = map(list, zip(*player_lag_grads_and_vars))
            
            # Wrap as tf.constant for tfe compatibility, [0] to unpack from tuple                        
            return(tf.constant(grads[0]))

class Trainer(object):
    def __init__(self, config, mode, net, clip_op_lambda):
        self.config = config
        self.mode = mode
        # TODO: Factor Model Owner out of Trainer and instantiate seperately
        self.player_name='model-owner'
        # ISSUE?: During agent computation, player_name in build_update_steps is still model_owner.
        # This is confusing. This also happens in the tfe federated learning example.
        
        # Create output-dir
        if not os.path.exists(self.config.dir_name): os.mkdir(self.config.dir_name)

        if self.mode == "train":
            log_suffix = '_' + str(self.config.train.restore_iter) if self.config.train.restore_iter > 0 else ''
            self.log_fname = os.path.join(self.config.dir_name, 'train' + log_suffix + '.txt')
        else:
            log_suffix = "_iter_" + str(self.config.test.restore_iter) + "_m_" + str(self.config.test.num_misreports) + "_gd_" + str(self.config.test.gd_iter)
            self.log_fname = os.path.join(self.config.dir_name, "test" + log_suffix + ".txt")

        # Set Seeds for reproducibility
        np.random.seed(self.config[self.mode].seed)
        tf.set_random_seed(self.config[self.mode].seed)

        # Init Logger
        self.init_logger()

        # Init Net
        self.net = net

        ## Clip Op
        self.clip_op_lambda = clip_op_lambda

        # Init TF-graph
        self.init_graph()

    def __del__(self):
        self.logger.removeHandler(self.loghandler)
        self.logger.removeHandler(self.filehandler)

    # >>> Helper functions for Differential Privacy (DP)
    def gen_ledger(self, pop_size, batch_size):
        ledger = privacy_ledger.PrivacyLedger(
            population_size = pop_size,
            selection_probability = (batch_size / pop_size))
        return(ledger)
    # <<<
    
    def calc_priv(self, sess):
        tic = time.time()
        orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
        _samples, _queries = self.opt_3_ledger.get_unformatted_ledger()
        samples = sess.run(_samples)
        queries = sess.run(_queries)
        formatted_ledger = privacy_ledger.format_ledger(samples, queries)
        rdp = compute_rdp_from_ledger(formatted_ledger, orders)
        epsilon = get_privacy_spent(orders, rdp, target_delta=self.config.train.delta)[0]
        toc = time.time()
        priv_time = (toc-tic)
        return(epsilon, priv_time)

    # >>> Helper functions for Multiparty Computation (MPC)
    # TODO: Factor this out of Trainer
    def setup_agents(self, generator):
        self.agents = [ Agent("agent-" + str(i), self.build_update_step, generator) for i in range(0, self.config.num_agents) ]
    # <<<
        
    def get_clip_op(self, adv_var):
        self.clip_op =  self.clip_op_lambda(adv_var)
        #tf.assign(adv_var, tf.clip_by_value(adv_var, 0.0, 1.0))


    def init_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        self.loghandler = logging.StreamHandler()
        self.loghandler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        self.loghandler.setFormatter(formatter)
        logger.addHandler(self.loghandler)

        self.filehandler = logging.FileHandler(self.log_fname, 'w')
        self.filehandler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
        self.filehandler.setFormatter(formatter)
        logger.addHandler(self.filehandler)

        self.logger = logger

    def compute_rev(self, pay):
        """ Given payment (pay), computes revenue
            Input params:
                pay: [num_batches, num_agents]
            Output params:
                revenue: scalar
        """
        return tf.reduce_mean(tf.reduce_sum(pay, axis=-1))

    def compute_utility(self, x, alloc, pay):
        """ Given input valuation (x), payment (pay) and allocation (alloc), computes utility
            Input params:
                x: [num_batches, num_agents, num_items]
                a: [num_batches, num_agents, num_items]
                p: [num_batches, num_agents]
            Output params:
                utility: [num_batches, num_agents]
        """
        return tf.reduce_sum(tf.multiply(alloc, x), axis=-1) - pay


    def get_misreports(self, x, adv_var, adv_shape):
        num_misreports = adv_shape[1]
        adv = tf.tile(tf.expand_dims(adv_var, 0), [self.config.num_agents, 1, 1, 1, 1])
        x_mis = tf.tile(x, [self.config.num_agents * num_misreports, 1, 1])
        x_r = tf.reshape(x_mis, adv_shape)
        y = x_r * (1 - self.adv_mask) + adv * self.adv_mask
        misreports = tf.reshape(y, [-1, self.config.num_agents, self.config.num_items])
        return x_mis, misreports

    def init_graph(self):
        x_shape = [self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        adv_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        adv_var_shape = [ self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents, self.config.num_items]
        u_shape = [self.config.num_agents, self.config[self.mode].num_misreports, self.config[self.mode].batch_size, self.config.num_agents]

        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=x_shape, name='x')
        self.adv_init = tf.placeholder(tf.float32, shape=adv_var_shape, name='adv_init')

        self.adv_mask = np.zeros(adv_shape)
        self.adv_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents), :] = 1.0

        self.u_mask = np.zeros(u_shape)
        self.u_mask[np.arange(self.config.num_agents), :, :, np.arange(self.config.num_agents)] = 1.0

        with tf.variable_scope('adv_var'):
            self.adv_var = tf.get_variable('adv_var', shape = adv_var_shape, dtype = tf.float32)

        # Misreports
        x_mis, self.misreports = self.get_misreports(self.x, self.adv_var, adv_shape)

        # Get mechanism for true valuation: Allocation and Payment
        self.alloc, self.pay = self.net.inference(self.x)

        # Get mechanism for misreports: Allocation and Payment
        a_mis, p_mis = self.net.inference(self.misreports)

        # Utility
        utility = self.compute_utility(self.x, self.alloc, self.pay)
        utility_mis = self.compute_utility(x_mis, a_mis, p_mis)

        # Regret Computation
        u_mis = tf.reshape(utility_mis, u_shape) * self.u_mask
        utility_true = tf.tile(utility, [self.config.num_agents * self.config[self.mode].num_misreports, 1])
        excess_from_utility = tf.nn.relu(tf.reshape(utility_mis - utility_true, u_shape) * self.u_mask)
        rgt = tf.reduce_mean(tf.reduce_max(excess_from_utility, axis=(1, 3)), axis=1)

        #Metrics
        revenue = self.compute_rev(self.pay)
        rgt_mean = tf.reduce_mean(rgt)
        irp_mean = tf.reduce_mean(tf.nn.relu(-utility))

        # Variable Lists
        alloc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='alloc')
        pay_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='pay')
        var_list = alloc_vars + pay_vars

        if self.mode is "train":
            w_rgt_init_val = 0.0 if "w_rgt_init_val" not in self.config.train else self.config.train.w_rgt_init_val

            with tf.variable_scope('lag_var'):
                self.w_rgt = tf.Variable(np.ones(self.config.num_agents).astype(np.float32) * w_rgt_init_val, 'w_rgt')

            update_rate = tf.Variable(self.config.train.update_rate, trainable = False)
            self.increment_update_rate = update_rate.assign(update_rate + self.config.train.up_op_add)

            # Loss Functions
            rgt_penalty = update_rate * tf.reduce_sum(tf.square(rgt)) / 2.0
            lag_loss = tf.reduce_sum(self.w_rgt * rgt)

            loss_1 = -revenue + rgt_penalty + lag_loss
            loss_2 = -tf.reduce_sum(u_mis)

            # Vectorize loss_3 for usage with DP optimizer
            if (self.config.train.noise_multiplier, self.config.train.l2_norm_clip) != (None, None):
                loss_3 = [ -lag_loss ]
            else:
                loss_3 = -lag_loss

            reg_losses = tf.get_collection('reg_losses')
            if len(reg_losses) > 0:
                reg_loss_mean = tf.reduce_mean(reg_losses)
                loss_1 = loss_1 + reg_loss_mean

            learning_rate = tf.Variable(self.config.train.learning_rate, trainable = False)

            # Optimizer
            opt_1 = tf.train.AdamOptimizer(learning_rate)
            opt_2 = tf.train.AdamOptimizer(self.config.train.gd_lr)

            if (self.config.train.noise_multiplier, self.config.train.l2_norm_clip) != (None, None):
              # Population size is the number of agents
              self.opt_3_ledger = self.gen_ledger(self.config.train.pop_size, self.config.train.dp_batch_size)
              # TODO: Do we need the Gaussian version here?
              opt_3 = dp_optimizer.DPGradientDescentGaussianOptimizer(
                l2_norm_clip=self.config.train.l2_norm_clip,
                noise_multiplier=self.config.train.noise_multiplier,
                num_microbatches=self.config.train.microbatches,
                ledger=self.opt_3_ledger,
                learning_rate=update_rate)

            else:
              opt_3 = tf.train.GradientDescentOptimizer(update_rate)

            #  ops
            self.train_op = opt_1.minimize(loss_1, var_list = var_list)
            self.train_mis_step = opt_2.minimize(loss_2, var_list = [self.adv_var])

            if self.config.mpc:
                self.compute_lagrange_update = opt_3.compute_gradients(loss_3, var_list = [self.w_rgt])
                self.opt_3 = opt_3                
            else:
                self.lagrange_update = opt_3.minimize(loss_3, var_list = [self.w_rgt])

            print("w_rgt: ")
            print(self.w_rgt)

            # Val ops
            val_mis_opt = tf.train.AdamOptimizer(self.config.val.gd_lr)
            self.val_mis_step = val_mis_opt.minimize(loss_2, var_list = [self.adv_var])

            # Reset ops
            self.reset_train_mis_opt = tf.variables_initializer(opt_2.variables())
            self.reset_val_mis_opt = tf.variables_initializer(val_mis_opt.variables())

            # Metrics
            self.metrics = [revenue, rgt_mean, rgt_penalty, lag_loss, loss_1, tf.reduce_mean(self.w_rgt), update_rate]
            self.metric_names = ["Revenue", "Regret", "Reg_Loss", "Lag_Loss", "Net_Loss", "w_rgt_mean", "update_rate"]

            #Summary
            tf.summary.scalar('revenue', revenue)
            tf.summary.scalar('regret', rgt_mean)
            tf.summary.scalar('reg_loss', rgt_penalty)
            tf.summary.scalar('lag_loss', lag_loss)
            tf.summary.scalar('net_loss', loss_1)
            tf.summary.scalar('w_rgt_mean', tf.reduce_mean(self.w_rgt))
            if len(reg_losses) > 0: tf.summary.scalar('reg_loss', reg_loss_mean)

            self.merged = tf.summary.merge_all()
            self.saver = tf.train.Saver(max_to_keep = self.config.train.max_to_keep)

            #Save data

            self.train_val_columns = ["Player", "Iter", "Noise", "Clip", "Priv_Calc_Time", "Epsilon", "Train_Time"] + self.metric_names
            self.train_array = []
            self.val_array = []

        elif self.mode is "test":
            loss = -tf.reduce_sum(u_mis)
            test_mis_opt = tf.train.AdamOptimizer(self.config.test.gd_lr)
            self.test_mis_step = test_mis_opt.minimize(loss, var_list = [self.adv_var])
            self.reset_test_mis_opt = tf.variables_initializer(test_mis_opt.variables())

            # Metrics
            welfare = tf.reduce_mean(tf.reduce_sum(self.alloc * self.x, axis = (1,2)))
            self.metrics = [revenue, rgt_mean, irp_mean]
            self.metric_names = ["Revenue", "Regret", "IRP"]
            self.saver = tf.train.Saver(var_list = var_list)

            # Noise and Clip do not get used by test mode, but are needed to sort the data
            self.test_columns = ["Iter", "Noise", "Clip", "Batch", "Time", "Revenue", "Regret", "IRP"]
            self.test_array = []

        # Helper ops post GD steps
        self.assign_op = tf.assign(self.adv_var, self.adv_init)
        self.get_clip_op(self.adv_var)

    def init_train(self, generator, sess):
        """
        Runs training
        """
        # TODO: So far only ModelOwner loads and saves model. Have Agents save their own progress.
        # TODO: Distribute generator
        self.train_gen, self.val_gen = generator

        iter = self.config.train.restore_iter

        tf.global_variables_initializer().run()
        self.train_writer = tf.summary.FileWriter(self.config.dir_name, sess.graph)

        # ModelOwner does loading and storing
        if iter > 0:
            model_path = os.path.join(self.config.dir_name, 'model-' + str(iter))
            self.saver.restore(sess, model_path)

        if iter == 0:
            self.train_gen.save_data(0)

            # ISSUE with tfe.Session(): 'str' object has no attribute 'to_native'
            # self.saver.save(sess, os.path.join(self.config.dir_name, 'model'), global_step = iter)

        self.time_elapsed = 0.0

        return(iter)

    def build_update_step(self, iter, sess):
        # Loop until lagrange update
        while(True):
            # INFO: Until now we do not have any guarantees that the
            # model was correctly executed by each DataOwner since
            # only gradient aggregation is done with MPC. At this
            # point we still trust Agents to correctly apply DP. This
            # will be adressed in further iterations.

            # TODO: Do we need to update model parameters globally or
            # can each player just compute locally? In this case, only
            # aggregated lagrange multipliers would become
            # public. Maybe only update alloc and payment parameters
            # once every lagrange update? How does this affect model
            # convergence and privacy leakage?

            # TODO: Different namespaces/scopes for different agents?
            
            X, ADV, perm = next(self.train_gen.gen_func)

            tic = time.time()

            # Get Best Mis-report
            sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})
            for i in range(self.config.train.gd_iter):
                sess.run(self.train_mis_step, feed_dict = {self.x: X})
                sess.run(self.clip_op)
            sess.run(self.reset_train_mis_opt)

            if self.config.train.data is "fixed" and self.config.train.adv_reuse:
                self.train_gen.update_adv(perm, sess.run(self.adv_var))

            # Update network params
            sess.run(self.train_op, feed_dict = {self.x: X})

            if iter==0:
                summary = sess.run(self.merged, feed_dict = {self.x: X})
                self.train_writer.add_summary(summary, iter)

            # Do _NOT_ return iter and increment it globally after each
            # player has computed gradients. The following counter is per Agent.
            # TODO: Replace with a tfe or tff idiom.
            iter += 1

            #if iter % 10 == 0:
            #    print("iter: " + str(iter))
                        
            # Compute Lagrange gradients
            if iter % self.config.train.update_frequency == 0:
                # ISSUE with tfe.Session(): 'RefVariable' object has no attribute 'to_native'
                # compute_gradients returns a list of (gradient, variable)
                player_lag_grads_and_vars = sess.run(self.compute_lagrange_update, feed_dict = {self.x: X})

            # TODO: Can this be moved to model owner computation?
            if iter % self.config.train.up_op_frequency == 0:
                sess.run(self.increment_update_rate)

            # TODO: Measure time for Data Owners, Model Owners and the whole run
            toc = time.time()
            self.time_elapsed += (toc - tic)

            # LOGGING TODO: Improve state passing for log or call object fields 
            # LOGGING TODO: Only log on player devices
            # self.train_log(iter, session, X, self.time_elapsed)

            if iter % self.config.train.update_frequency == 0:
                return(player_lag_grads_and_vars)


    @tfe.local_computation(player_name='model-owner')
    def update_model(self, aggregated_grads, iter, session):
        sess = session

        # Evaluate for compatibility with apply_gradients
        grads = aggregated_grads.eval()
        
        # TODO: Should we instead carry the vars from the users with us?
        # TODO: (un)packing helper functions
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lag_var')
        packed_grads = [np.array(grads, dtype=np.float32)]
        packed_grads_and_vars = list(zip(packed_grads, variables))

        apply_op = self.opt_3.apply_gradients(packed_grads_and_vars)

        return(apply_op)

    def train_save(self, iter, sess):
        if ((iter % self.config.train.save_iter) == 0) or (iter == self.config.train.max_iter):
            self.saver.save(sess, os.path.join(self.config.dir_name,'model'), global_step = iter)
            self.train_gen.save_data(iter)

    # LOGGING TODO: Factor out logging for agents and model owner.
    #               Build logging for dev (aggregate logs) and production debug (only locally)
    def train_log(self, iter, sess, X, time_elapsed):
        if (iter % self.config.train.print_iter) == 0:
            # Train Set Stats
            summary = sess.run(self.merged, feed_dict = {self.x: X})
            self.train_writer.add_summary(summary, iter)
            metric_vals = sess.run(self.metrics,  feed_dict = {self.x: X})
            fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_vals) for item in tup ])
            log_str = "TRAIN-BATCH Player: %s, Iter: %d, t = %.4f"%(self.player_name, iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
            self.logger.info(log_str)

            if (self.config.train.noise_multiplier, self.config.train.l2_norm_clip) != (None,None):
              eps, priv_time = self.calc_priv(sess)
              priv_str = "epsilon: %.2f, calculation time: %.2f, noise: %.5f, clip: %.5f"%(eps, priv_time, self.config.train.noise_multiplier, self.config.train.l2_norm_clip)
              self.logger.info(priv_str)
              train_row = [self.player_name, iter, self.config.train.noise_multiplier, self.config.train.l2_norm_clip, priv_time, eps, time_elapsed] + metric_vals

            else:
              # When running an experiment without DP, save all DP measurements as 0.
              train_row = [self.player_name, iter, 0, 0, 0, 0, time_elapsed] + metric_vals

            self.train_array.append(train_row)
            # LOGGING TODO: Save .csv per player, on player machines, adapt 
            #pd.DataFrame(self.train_array, columns=self.train_val_columns).to_csv(os.path.join(self.config.dir_name,self.player_name+'_train_data.csv'))
            pd.DataFrame(self.train_array, columns=self.train_val_columns).to_csv(os.path.join(self.config.dir_name,'train_data.csv'))


        if (iter % self.config.val.print_iter) == 0:
            #Validation Set Stats
            metric_tot = np.zeros(len(self.metric_names))
            for _ in range(self.config.val.num_batches):
                X, ADV, _ = next(self.val_gen.gen_func)
                sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})
                for k in range(self.config.val.gd_iter):
                    sess.run(self.val_mis_step, feed_dict = {self.x: X})
                    sess.run(self.clip_op)
                sess.run(self.reset_val_mis_opt)
                metric_vals = sess.run(self.metrics, feed_dict = {self.x: X})
                metric_tot += metric_vals

            metric_tot = metric_tot/self.config.val.num_batches
            fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
            log_str = "VAL-%d"%(iter) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
            self.logger.info(log_str)

            if (self.config.train.noise_multiplier, self.config.train.l2_norm_clip) != (None,None):
              eps, priv_time = self.calc_priv(sess)
              priv_str = "epsilon: %.2f, calculation time: %.2f, noise: %.5f, clip: %.5f"%(eps, priv_time, self.config.train.noise_multiplier, self.config.train.l2_norm_clip)
              self.logger.info(priv_str)
              val_row = [self.player_name, iter, self.config.train.noise_multiplier, self.config.train.l2_norm_clip, priv_time, eps, time_elapsed] + metric_vals

            else:
              # When running an experiment without DP, save all DP measurements as 0.
              val_row = [self.player_name, iter, 0, 0, 0, 0, time_elapsed] + metric_vals

            self.val_array.append(val_row)
            #pd.DataFrame(self.val_array, columns=self.train_val_columns).to_csv(os.path.join(self.config.dir_name,self.player_name+'_val_data.csv'))
            pd.DataFrame(self.val_array, columns=self.train_val_columns).to_csv(os.path.join(self.config.dir_name,'val_data.csv'))



    def train(self, generator):
        config = tf.ConfigProto(
            device_count = {'CPU': self.config.num_agents + 5} # TODO: Model- + DataOwners + Master + 2 MPC?
        )
        # TODO: Get tfe.Session() to work
        session = tf.InteractiveSession(config=config)
        
        #with tf.device(tfe.get_config().get_player(self.player_name).device_name):
        print("INIT TRAIN")
        iter = self.init_train(generator, session)

        self.setup_agents(generator)

        time_elapsed = 0
        
        while iter < (self.config.train.max_iter):
            # TODO: For now only lagrangian updates, distribution of allocation and payment
            #       computation will come later
            tic = time.time()
            #print("global iter pre: " + str(iter))
            
            if iter % self.config.train.update_frequency == 0:
                all_lag_grads = [
                    agent.compute_grads(iter, session)
                    for agent in self.agents
                ]                
                aggregated_grads = tfe.add_n(all_lag_grads) / int(all_lag_grads[0].shape[0])
            
                apply_op = self.update_model(aggregated_grads, iter, session)
                session.run(apply_op)
                
            # Increase iter by # of steps in the Agent loops. This has to happen at the end of this loop.
            iter += self.config.train.update_frequency
            #print("global iter post: " + str(iter))

            toc = time.time()
            time_elapsed += (toc - tic)

            # Sample a fresh X to compute data for logging purposes only.
            # This is a rough approximation.
            # LOGGING TODO: After logging is refactored, find a more accurate solution.
            X, ADV, perm = next(self.train_gen.gen_func)

            self.train_log(iter, session, X, time_elapsed)
            self.train_save(iter, session)

    def test(self, generator):
        """
        Runs test
        """
        # LOGGING TODO: Read playerwise .csv files
        # TODO: Playerwise tests?
        
        # Init generators
        self.test_gen = generator

        iter = self.config.test.restore_iter
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        model_path = os.path.join(self.config.dir_name,'model-' + str(iter))
        self.saver.restore(sess, model_path)

        #Test-set Stats
        time_elapsed = 0

        metric_tot = np.zeros(len(self.metric_names))

        if self.config.test.save_output:
            assert(hasattr(generator, "X")), "save_output option only allowed when config.test.data = Fixed or when X is passed as an argument to the generator"
            alloc_tst = np.zeros(self.test_gen.X.shape)
            pay_tst = np.zeros(self.test_gen.X.shape[:-1])

        for i in range(self.config.test.num_batches):
            tic = time.time()
            X, ADV, perm = next(self.test_gen.gen_func)
            sess.run(self.assign_op, feed_dict = {self.adv_init: ADV})

            for k in range(self.config.test.gd_iter):
                sess.run(self.test_mis_step, feed_dict = {self.x: X})
                sess.run(self.clip_op)

            sess.run(self.reset_test_mis_opt)

            metric_vals = sess.run(self.metrics, feed_dict = {self.x: X})

            if self.config.test.save_output:
                A, P = sess.run([self.alloc, self.pay], feed_dict = {self.x:X})
                alloc_tst[perm, :, :] = A
                pay_tst[perm, :] = P

            metric_tot += metric_vals
            toc = time.time()
            time_elapsed += (toc - tic)

            fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_vals) for item in tup ])
            log_str = "TEST BATCH-%d: t = %.4f"%(i, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
            self.logger.info(log_str)

        metric_tot = metric_tot/self.config.test.num_batches
        fmt_vals = tuple([ item for tup in zip(self.metric_names, metric_tot) for item in tup ])
        log_str = "TEST ALL-%d: t = %.4f"%(iter, time_elapsed) + ", %s: %.6f"*len(self.metric_names)%fmt_vals
        self.logger.info(log_str)
        test_row = [iter, self.config.train.noise_multiplier, self.config.train.l2_norm_clip, i, time_elapsed] + metric_vals
        self.test_array.append(test_row)
        pd.DataFrame(self.test_array, columns=self.test_columns).to_csv(os.path.join(self.config.dir_name, 'iter_' + str(iter) + '_test_data.csv'))

        if self.config.test.save_output:
            np.save(os.path.join(self.config.dir_name, 'alloc_tst_' + str(iter)), alloc_tst)
            np.save(os.path.join(self.config.dir_name, 'pay_tst_' + str(iter)), pay_tst)

        sess.close()
