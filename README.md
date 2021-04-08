# Prior-Free One-Shot Auctions via Differential Privacy
Preprint: [Towards Prior-Free Approximately Truthful One-Shot Auction Learning via Differential Privacy](https://arxiv.org/abs/2104.00159).

This is a fork of [Optimal Auctions through Deep Learning](https://github.com/saisrivatsan/deep-opt-auctions) (https://arxiv.org/pdf/1706.03459.pdf), using [TensorFlow Privacy](https://github.com/tensorflow/privacy/) to make the RegretNet approach Differentially Private, resulting in Approximate Truthfulness and Collusion Resistance in the sense of [Mechanism Design via Differential Privacy](http://kunaltalwar.org/papers/expmech.pdf). This means we can relax the assumption of having prior knowledge of the valuation profiles. 
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
- [Sample of 128 valuation sets, showing bounded regret/utility of misreporting agent](https://nbviewer.jupyter.org/github/degregat/deep-opt-auctions/blob/sample_128_5x3/regretNet/batch_experiments/additive_5x3_exh_1/visualize_valuation_sample_128_5x3.ipynb)
([data](https://github.com/degregat/deep-opt-auctions/blob/sample_128_5x3/regretNet/batch_experiments/additive_5x3_exh_1/))

- [Different sample of size 128. Covers a wider range of outcomes, used in arXiv submission](https://nbviewer.jupyter.org/github/degregat/one-shot-approx-auctions/blob/arXiv-submission-data/regretNet/batch_experiments/additive_5x3_exh_1/visualize_valuation_sample.ipynb)
([data and figures](https://github.com/degregat/one-shot-approx-auctions/blob/arXiv-submission-data/regretNet/batch_experiments/additive_5x3_exh_1/))

- [Multiple runs from a single valuation set, showing non-determinism of training](https://nbviewer.jupyter.org/github/degregat/deep-opt-auctions/blob/exp_one_shot_single_valuation_1/regretNet/batch_experiments/one_shot_single_valuation_1/visualize_one_shot_comparison.ipynb)
([data](https://github.com/degregat/deep-opt-auctions/blob/exp_one_shot_single_valuation_1/regretNet/batch_experiments/one_shot_single_valuation_1/))

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
