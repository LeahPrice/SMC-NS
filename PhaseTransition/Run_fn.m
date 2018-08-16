%clear all

function Run_fn(N,R,runs,exact)
options.N = N; % number of particles
options.R = R; % repeats for MCMC steps
options.sig = 0.1; % for MCMC steps
options.d = 10; % dimension

options.c = 0*ones(1,options.d);

options.alpha = 0.95;
options.rho = exp(-1); % for SMC-NS
options.stopping_propZ = 0.999; % SMC-NS stopping rule
options.stopping_number = inf; % SMC-NS stopping rule
options.stopping_ESS = inf; % SMC-NS stopping rule
options.tol = 0.01; % for NS tuning of levels

vol = pi^(options.d/2)/factorial(options.d/2); % normalising constant of the prior
seed = 12345;
rng(seed);
ell = zeros(runs,1);

options.R = 10;

%Nested Sampling / Improved Nested Sampling
ell3 = zeros(runs,1); ell3_adj = zeros(runs,1);
evals = 0;
fprintf('\nNS Runs: \n')
for i = 1:runs
    fprintf('%.0f ', i)
    if exact
        [theta, log_weights, log_evidence, count_loglike, log_adj] = NS_exact(options);
    else
        [theta, log_weights, log_evidence, count_loglike, log_adj] = NS_MCMC(options);
    end
    
    ell3(i) = exp(log_evidence)*vol;
    ell3_adj(i) = exp(log_adj)*vol;
    evals = evals + count_loglike;
end


fprintf('\nNS: %.3f, std: %.2e, avg. evals: %.0f', mean(ell3), std(ell3)/sqrt(runs), evals/runs)
fprintf('\nINS: %.3f, std: %.2e, avg. evals: %.0f\n', mean(ell3_adj), std(ell3_adj)/sqrt(runs), evals/runs)


rng(seed)
evals = 0;
fprintf('\nSMC-NS Adaptive Runs: \n')

for i = 1:runs
    fprintf('%.0f ', i)
    if exact
        [theta, log_weights, log_evidence, count_loglike, levels, distances] = NSSMC_exact_adaptive(options);
    else
        [theta, log_weights, log_evidence, count_loglike, levels] = NSSMC_MCMC_adaptive(options);
    end
    ell(i) = exp(log_evidence)*vol;
    evals = evals + count_loglike;
end

fprintf('\n\nSMC-NS Adaptive: %.3f, std: %.2e, avg. evals: %.0f \n', mean(ell), std(ell)/sqrt(runs), evals/runs)


%SMC-NS: fixed run
    fprintf('\n\nSMC-NS Fixed Runs: \n')
    options.levels = levels;
  
    
    ell2 = zeros(runs,1);
    evals =0;
    for i = 1:runs
        fprintf('%.0f ', i)
        if exact
            options.distances = distances;
            [theta, log_weights, log_evidence, count_loglike, error_flag] = NSSMC_exact_fixed(options);
        else
            [theta, log_weights, log_evidence, count_loglike, error_flag] = NSSMC_MCMC_fixed(options);
        end
        ell2(i) =exp(log_evidence)*vol;
        evals = evals + count_loglike;
    end

    fprintf('\n\nNS-SMC Fixed: %.3f, std: %.2e, avg. evals: %.0f \n', mean(ell2), std(ell2)/sqrt(runs),  evals/runs)


% TA-SMC: adaptive run with MCMC sampler
ell_Anneal = zeros(runs,1);
options.R = 20;
evals = 0;
%fprintf('\nTA-SMC Runs: \n')
for i = 1:runs
    fprintf('%.0f ', i)
    [theta, loglike, log_evidence, count_loglike, gammavar] = anneal_MCMC_adaptive(options);
    ell_Anneal(i) = exp(log_evidence)*vol;
    evals = evals + count_loglike;
end
fprintf('\nTA-SMC: %.3f, std: %.2e, avg. evals: %.0f\n', mean(ell_Anneal), std(ell_Anneal)/sqrt(runs), evals/runs)



% boxplot
    

% clf
% boxplot([ell, ell2, ell3], 'Labels', {'Adaptive SMC-NS', 'SMC-NS',  'NS'})
% xtickangle(45)
% figure(1)
