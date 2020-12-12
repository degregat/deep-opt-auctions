# Approximately Optimal Auctions through Differentially Private One-Shot Learning
Fork of [Optimal Auctions through Deep Learning](https://github.com/saisrivatsan/deep-opt-auctions) (https://arxiv.org/pdf/1706.03459.pdf), using [TensorFlow Privacy](https://github.com/tensorflow/privacy/) to make the RegretNet approach Differentially Private, resulting in Approximate Truthfulness and Collusion Resistance in the sense of [Mechanism Design via Differential Privacy](http://kunaltalwar.org/papers/expmech.pdf). This means we can relax the assumption of having prior knowledge of the valuation profiles.

Only a single report is taken from each agent. We perform one-shot learning on these to learn an auction and plug in those same reports to calculate Revenue, Regret and Welfare.

Since the potential to outperform a truthful report in the learned mechanism, by misreporting during the training, should be bounded, the incentive to misreport would also be bounded. Thus, the mechanism learner should be approximately truthful.

## Changes in the one-shot learner

The optimizer for the auction parameters (`opt_1` in the code) (see Section 4 of the [paper on "Optimal Auctions through Deep Learning"](https://arxiv.org/pdf/1706.03459.pdf)) in this fork is differentially private, bounding the influence of reports on the resulting auction. Therefore, the influence each agent has on the resulting allocation and payment functions is bounded.

## Analysis

As a first attempt to verify approximate truthfulness, we take one set of valuations [v_0, ..., v_N-1] for N agents.

We take  enumerate all possible (mis)reports [m_0, m_S-1] (sample size of S) for Agent_0, take a single misreport leaving the valuations of the other agents fixed ([m_0, v_1, ..., v_N-1] to [m_S-1, v_1, ..., v_N-1]) and train one auction per set of reports.

To empirically measure the performance, we compare utility, regret and welfare across all auctions learned from the (mis)report + valuations.

To explore the valuation space, we sample different valuation sets and repeat the above process.

### Preliminary results
In settings without differential privacy, the misreporting agent is able to outperform the others by a large margin (for some valuation sets).

In settings with differential privacy, the margin by which the misreporting agent can outperform the others gets bounded more tightly (for all sampled valuation sets).

As noise increases, we transition to a lottery, meaning the bounds on regret and utility for each agent are broadened. Revenue and welfare are decreased.

Example plots:
- [Multiple runs, single valuation set](https://github.com/degregat/deep-opt-auctions/blob/exp_one_shot_single_valuation_1/regretNet/batch_experiments/one_shot_single_valuation_1/visualize_one_shot_comparison.ipynb) (to show non-determinism from training)

### Limitations

For now, we only analyze agent types constrained to 0/1, which enables us to feasibly enumerate all possible (mis)reports for an agent up to the 10 item case (2^10 = 1024 misreports).

### TODO

- Quantify approximation of truthfulness in relation to epsilon for different settings
- Benchmark
  - against deep-opt-auctions
  - against e.g. second price auction

## Getting Started

- Install Python 3.7
- clone this repository and `cd` into it
- run `pip3 install -r requirements.txt`

## Running the experiments

### RegretNet

in `deep-opt-auctions/regretNet` execute `./run_batch.py`

Supported settings so far are `additive_1x2, additive_5x3, additive_5x10` (which are additive_NxM_uniform, but with types constrained to 0/1).

Basic example:
`./run_batch.py --setting additive_5x3 --noise-vals 0.03 0.05 0.09 --clip-vals 1 --add-no-dp-run --iterations 500 --pool-size 16 --sample-size 128`

If you want to learn another set of auctions (since learning is not deterministic) for the same valuation sets, using up to 16 processes in parallel:
`./run_batch.py --setting additive_5x3 --noise-vals 0.03 --clip-vals 1 --iterations 500 --pool-size 16 --valuations some/valuations.npy`









## What follows is the old README.md

# Optimal Auctions through Deep Learning
Implementation of "Optimal Auctions through Deep Learning" (https://arxiv.org/pdf/1706.03459.pdf)

## Getting Started

Install the following packages:
- Python 2.7
- Tensorflow
- Numpy and Matplotlib packages
- Easydict - `pip install easydict`

## Running the experiments

### RegretNet

#### For Gradient-Based approach:
Default hyperparameters are specified in regretNet/cfgs/.

#### For Sample-Based approach:
Modify the following hyperparameters in the config file specified in regretNet/cfg/.
```
cfg.train.gd_iter = 0
cfg.train.num_misreports = 100
cfg.val.num_misreports = 100 # Number of val-misreports is always equal to the number of train-misreports
```

For training the network, testing the mechanism learnt and computing the baselines, run:
```
cd regretNet
python run_train.py [setting_name]
python run_test.py [setting_name]
python run_baseline.py [setting_name]
```

setting\_no  |      setting\_name |
 :---:   | :---: |
  (a)    |  additive\_1x2\_uniform |
  (b)   | unit\_1x2\_uniform\_23 |
  (c\)  | additive\_2x2\_uniform |
  (d)   | CA\_sym\_uniform\_12 |
  (e)    | CA\_asym\_uniform\_12\_15 |
  (f)   | additive\_3x10\_uniform |
  (g)  | additive\_5x10\_uniform |
  (h) |   additive\_1x2\_uniform\_416\_47
  (i) |   additive\_1x2\_uniform\_triangle
  (j) |   unit\_1x2\_uniform
  (k) |  additive\_1x10\_uniform
  (l) |   additive\_1x2\_uniform\_04\_03
  (m) |   unit\_2x2\_uniform


### RochetNet (Single Bidder Auctions)

Default hyperparameters are specified in rochetNet/cfgs/.
For training the network, testing the mechanism learnt and computing the baselines, run:
```
cd rochetNet
python run_train.py [setting_name]
python run_test.py [setting_name]
python run_baseline.py [setting_name]
```
setting\_no  |      setting\_name |
 :---:  | :---: |
  (a)   |  additive\_1x2\_uniform |
  (b)   |   additive\_1x2\_uniform\_416\_47
  \(c\) |   additive\_1x2\_uniform\_triangle
  (d)   |   additive\_1x2\_uniform\_04\_03
  (e)   |  additive\_1x10\_uniform
  (f)   |   unit\_1x2\_uniform
  (g)   |   unit\_1x2\_uniform\_23

### MyersonNet (Single Item Auctions)

Default hyperparameters are specified in utils/cfg.py.
For training the network, testing the mechanism learnt and computing the baselines, run:
```
cd myersonNet
python main.py -distr [setting_name] or
bash myerson.sh
```
setting\_no  |      setting\_name |
 :---:  | :---: |
  (a)   |  exponential
  (b)   |   uniform
  \(c\) |   asymmetric\_uniform
  (d)   |   irregular


## Settings

### Single Bidder
- **additive\_1x2\_uniform**: A single bidder with additive valuations over two items, where the items is drawn from U\[0, 1\].

- **unit\_1x2\_uniform\_23**: A single bidder with unit-demand valuations over two items, where the item values are drawn from U\[2, 3\].

- **additive\_1x2\_uniform\_416\_47**: Single additive bidder with preferences over two non-identically distributed items, where v<sub>1</sub> ∼ U\[4, 16\]and v<sub>2</sub> ∼ U\[4, 7\].

- **additive\_1x2\_uniform\_triangle**: A single additive bidder with preferences over two items, where (v<sub>1</sub>, v<sub>2</sub>) are drawn jointly and uniformly from a unit-triangle with vertices (0, 0), (0, 1) and (1, 0).

- **unit\_1x2\_uniform**: A single unit-demand bidder with preferences over two items, where the item values from U\[0, 1\]

- **additive\_1x2\_uniform\_04\_03**: A Single additive bidder with preferences over two items, where the item values v<sub>1</sub> ∼ U\[0, 4], v<sub>2</sub> ∼ U\[0, 3]

- **additive\_1x10\_uniform**: A single additive bidder and 10 items, where bidders draw their value for each item from U\[0, 1\].

### Multiple Bidders
- **additive\_2x2\_uniform**: Two additive bidders and two items, where bidders draw their value for each item from U\[0, 1\].

- **unit\_2x2\_uniform**: Two unit-demand bidders and two items, where the bidders draw their value for each item from identical U\[0, 1\].

- **additive\_2x3\_uniform**: Two additive bidders and three items, where bidders draw their value for each item from U\[0, 1\].

- **CA\_sym\_uniform\_12**: Two bidders and two items, with v<sub>1,1</sub>, v<sub>1,2</sub>, v<sub>2,1</sub>, v<sub>2,2</sub> ∼ U\[1, 2\], v<sub>1,{1,2}</sub> = v<sub>1,1</sub> + v<sub>1,2</sub> + C<sub>1</sub> and v<sub>2,{1,2}</sub> = v<sub>2,1</sub> + v<sub>2,2</sub> + C<sub>2</sub>, where C<sub>1</sub>, C<sub>2</sub> ∼ U\[−1, 1\].

- **CA\_asym\_uniform\_12\_15**: Two bidders and two items, with v<sub>1,1</sub>, v<sub>1,2</sub> ∼ U\[1, 2\], v<sub>2,1</sub>, v<sub>2,2</sub> ∼ U\[1, 5\], v<sub>1,{1,2}</sub> = v<sub>1,1</sub> + v<sub>1,2</sub> + C<sub>1</sub> and v<sub>2,{1,2}</sub> = v<sub>2,1</sub> + v<sub>2,2</sub> + C<sub>2</sub>, where C<sub>1</sub>, C<sub>2</sub> ∼ U\[−1, 1].

- **additive\_3x10\_uniform**: 3 additive bidders and 10 items, where bidders draw their value for each item from U\[0, 1\].

- **additive\_5x10\_uniform**: 5 additive bidders and 10 items, where bidders draw their value for each item from U\[0, 1\].


## Visualization

Allocation Probabilty plots for **unit\_1x2\_uniform_23** setting learnt by **regretNet**:

<img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/regretNet/plots/visualization/unit_1x2_uniform_23_alloc1.png" width="300"> <img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/regretNet/plots/visualization/unit_1x2_uniform_23_alloc2.png" width="300">

Allocation Probabilty plots for **additive\_1x2\_uniform\_416\_47** setting learnt by **rochetNet**:

<img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/rochetNet/plots/visualization/additive_1x2_uniform_416_47_alloc1.png" width="300"> <img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/rochetNet/plots/visualization/additive_1x2_uniform_416_47_alloc2.png" width="300">

For other allocation probability plots, check-out the ipython notebooks in `regretNet` or `rochetNet` folder.


## Reference

Please cite our work if you find our code/paper is useful to your work.
```
@article{DFNP19,
  author    = {Paul D{\"{u}}tting and Zhe Feng and Harikrishna Narasimhan and David C. Parkes and Sai Srivatsa Ravindranath},
  title     = {Optimal Auctions through Deep Learning},
  journal   = {arXiv preprint arXiv:1706.03459},
  year      = {2019},
}
```
