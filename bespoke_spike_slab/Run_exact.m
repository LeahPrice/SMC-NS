N = 100; % Run for 100, 1000 and 10000
d = 10;
alpha = exp(-1);

if (N==100)
    num_repeats = 10000;
elseif N==1000
    num_repeats = 1000;
elseif N==10000
    num_repeats = 100;
end

log_evidence_ans_smc = NaN(num_repeats,1);
count_ans_smc = NaN(num_repeats,1);

log_evidence_ns_smc = NaN(num_repeats,1);
count_ns_smc = NaN(num_repeats,1);
error_flag_ns_smc = NaN(num_repeats,1);

log_evidence_ns = NaN(num_repeats,1);
count_ns = NaN(num_repeats,1);

log_evidence_ns_star = NaN(num_repeats,1);
count_ns_star = NaN(num_repeats,1);

for i=1:num_repeats

    %%%%%%%%%%%%%%%%%%%% PERFORMING ADAPTIVE NS-SMC %%%%%%%%%%%%%%%%%%%%%%%

    % ANS-SMC
    options = [];
    options.N = N;
    options.d = d;
    options.alpha = alpha;
    
    rng(i);
    [log_evidence, count_loglike, levels, dists] = ANS_SMC_bespoke_exact(@loglike_fn,@simprior_fn,options,false);

    log_evidence_ans_smc(i) = log_evidence;
    count_ans_smc(i) = count_loglike;

    %%%%%%%%%%%%%%%%%%%%%%%% PERFORMING FIXED NS-SMC %%%%%%%%%%%%%%%%%%%%%%

    % NS-SMC RW
    options.N = N;
    options.d = d;
    options.levels = levels;
    options.dists = dists;

    rng(i+num_repeats);
    [log_evidence, count_loglike, error_flag] = NS_SMC_bespoke_exact(@loglike_fn,@simprior_fn,options,false);
    log_evidence_ns_smc(i) = log_evidence;
    count_ns_smc(i) = count_loglike + count_ans_smc(i);
    error_flag_ns_smc(i) = error_flag;

    %%%%%%%%%%%%%%%%% PERFORMING NS (WITH MAX EVALS) %%%%%%%%%%%%%%%%%%%%%%

    % NS RW
    options = [];
    options.N = N;
    options.d = d;
    options.desired_count = count_ns_smc(i);

    rng(i);
    [log_evidence, log_evidence_star, count_loglike] = NS_bespoke_exact(@loglike_fn,@simprior_fn,options,false);

    log_evidence_ns(i) = log_evidence;
    count_ns(i) = count_loglike;
    log_evidence_ns_star(i) = log_evidence_star;
    count_ns_star(i) = count_loglike;

    if (mod(i,num_repeats/100)==0)
        fprintf('Iteration %d.\n', i);
    end
end

clearvars -except N d alpha stopping_epsilon log_evidence_* count_* error_flag_* num_repeats;
clear log_evidence_star;

filename = sprintf('results/SpikeSlab_exact_N%d.mat', N); save(filename);
