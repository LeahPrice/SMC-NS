# NS-SMC
This repository provides code accompanying the paper "Unbiased and consistent nested sampling via sequential Monte Carlo", available at https://arxiv.org/abs/1805.03924.

The folder *code* is a library for general use. To use this for a new example, you will need
- A list object (referred to as `options` in the code) that includes the required algorithm hyperparameters (i.e. N, alpha, etc) and data.
- Functions `loglike_fn` and `logprior_fn` that evaluate the log likelihood and log prior, respectively, taking a sample and the `options` list as arguments.
- A `simprior_fn` function that simulates `N` samples from the prior, taking `N` and the `options` as input.

# Reproducing the factor analysis results

The factor analysis example uses the library in the code folder. You can recreate the results files in FA/results (and another 1500 files required for creating Figure 3) by running FA/Run.m and then FA/results/combine_results.m. There are files for reproducing the information in Table 3, Figure 2 and Figure 3.

# Reproducing the spike-and-slab results

The spike-and-slab example uses bespoke samplers so we do not use the library in the code folder. You can recreate the results files in bespoke_spike_slab/results by running bespoke_spike_slab/Run_exact.m and bespoke_spike_slab/Run_RW.m. There is a script bespoke_spike_slab/results/Table2.m for reproducing the information in Table 2.
