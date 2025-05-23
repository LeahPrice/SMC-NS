N = 100; % Run for 100, 1000 and 10000
d = 10;
alpha = exp(-1);
repeats = 10;

if (N==100)
    num_repeats = 10000;
elseif N==1000
    num_repeats = 1000;
elseif N==10000
    num_repeats = 100;
end

log_evidence_ata_smc = NaN(num_repeats,1);
count_ata_smc = NaN(num_repeats,1);
time_ata_smc = NaN(num_repeats,1);

log_evidence_ta_smc = NaN(num_repeats,1);
count_ta_smc = NaN(num_repeats,1);
time_ta_smc = NaN(num_repeats,1);

log_evidence_ans_smc = NaN(num_repeats,1);
count_ans_smc = NaN(num_repeats,1);
time_ans_smc = NaN(num_repeats,1);

log_evidence_ns_smc = NaN(num_repeats,1);
count_ns_smc = NaN(num_repeats,1);
time_ns_smc = NaN(num_repeats,1);
error_flag_ns_smc = NaN(num_repeats,1);

log_evidence_ns = NaN(num_repeats,1);
count_ns = NaN(num_repeats,1);
time_ns = NaN(num_repeats,1);

log_evidence_ns_star = NaN(num_repeats,1);
count_ns_star = NaN(num_repeats,1);
time_ns_star = NaN(num_repeats,1);

for i=1:num_repeats

    %%%%%%%%%%%%%%%%%%%% PERFORMING ADAPTIVE TA-SMC %%%%%%%%%%%%%%%%%%%%%%%

    % ATA-SMC
    options = [];
    options.N = N;
    options.d = d;
    options.alpha = 0.999;
    options.R = repeats;
    options.sig = 0.1;

    rng(i);
    tic; [log_evidence, count_loglike, gammavar] = ATA_SMC_bespoke_RW(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;

    log_evidence_ata_smc(i) = log_evidence;
    count_ata_smc(i) = count_loglike;
    time_ata_smc(i) = time;

    %%%%%%%%%%%%%%%%%%%%%%% PERFORMING FIXED TA-SMC %%%%%%%%%%%%%%%%%%%%%%%

    % TA-SMC
    options.N = N;
    options.d = d;
    options.gammavar = gammavar;
    options.R = repeats;
    options.sig = 0.1;

    rng(i+num_repeats);
    tic; [log_evidence, count_loglike] = TA_SMC_bespoke_RW(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;
    log_evidence_ta_smc(i) = log_evidence;
    count_ta_smc(i) = count_loglike + count_ata_smc(i);
    time_ta_smc(i) = time + time_ata_smc(i);

    %%%%%%%%%%%%%%%%%%%%%% PERFORMING ADAPTIVE NS-SMC %%%%%%%%%%%%%%%%%%%%%

    % ANS-SMC
    options = [];
    options.N = N;
    options.d = d;
    options.alpha = alpha;
    options.R = repeats;
    options.sig = 0.1;

    rng(i);
    tic; [log_evidence, count_loglike, levels] = ANS_SMC_bespoke_RW(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;

    log_evidence_ans_smc(i) = log_evidence;
    count_ans_smc(i) = count_loglike;
    time_ans_smc(i) = time;

    %%%%%%%%%%%%%%%%%%%%%%%% PERFORMING FIXED NS-SMC %%%%%%%%%%%%%%%%%%%%%%

    % NS-SMC
    options.N = N;
    options.d = d;
    options.levels = levels;
    options.R = repeats;
    options.sig = 0.1;

    rng(i+num_repeats);
    tic; [log_evidence, count_loglike, error_flag] = NS_SMC_bespoke_RW(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;
    log_evidence_ns_smc(i) = log_evidence;
    count_ns_smc(i) = count_loglike + count_ans_smc(i);
    time_ns_smc(i) = time + time_ans_smc(i);
    error_flag_ns_smc(i) = error_flag;

    %%%%%%%%%%%%%%%%% PERFORMING NS (WITH MAX EVALS) %%%%%%%%%%%%%%%%%%%%%%

    % NS RW
    options = [];
    options.N = N;
    options.d = d;
    options.desired_count = count_ns_smc(i);
    options.R = repeats;
    options.sig = 0.1;

    rng(i);
    tic; [log_evidence, log_evidence_star, count_loglike] = NS_bespoke_RW(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;

    log_evidence_ns(i) = log_evidence;
    count_ns(i) = count_loglike;
    time_ns(i) = time;
    log_evidence_ns_star(i) = log_evidence_star;
    count_ns_star(i) = count_loglike;
    time_ns_star(i) = time;

    if (mod(i,num_repeats/100)==0)
        fprintf('Iteration %d.\n', i);
    end
end

clearvars -except N d alpha stopping_epsilon repeats log_evidence_* count_* time_* error_flag_* num_repeats;
clear log_evidence_star;

filename = sprintf('results/SpikeSlab_RW_N%d.mat', N); save(filename);
