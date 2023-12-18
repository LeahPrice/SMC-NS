% Organising file locations

p = fileparts(which('Run.m')); % getting the location of the 'Run.m' file
location = strcat(p(1:(strfind(p, '\FA'))), 'code'); % getting the location of the 'code' folder - Windows
% location = strcat(p(1:(strfind(p, '/FA'))), 'code'); % getting the location of the 'code' folder
storage_location = sprintf('%s/results',p);
addpath(genpath(location)); % adding the contents of the 'code' folder to the working directory

d = [12; 17; 21]; % dimensions for each model

% Running for all model choices and repeats

for k=1:300

    model_choice = floor((k - 1)/100) + 1;
    seed_adapt = mod(k,100);
    if seed_adapt == 0
        seed_adapt = 100;
    end
    seed_fixed = seed_adapt + 100;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ANS-SMC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Setting algorithm parameters
    options.N = 1000;
    options.alpha = exp(-1);
    options.stopping_epsilon = 10^(-5);
    options.model_choice = model_choice;
    options.h = 2.38/sqrt(d(model_choice));
    options.R = 10*model_choice;
    load('FA_data.mat'); options.y = data.y;

    % Running the algorithm
    clearvars -except d k model_choice seed* options storage_location; rng(seed_adapt);
    tic; [theta, log_weights, log_evidence, count_loglike, levels, cov_part, prod_c] = ANS_SMC(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;

    % Calculating KSD
    w = (exp(log_weights - logsumexp(log_weights)));
    inds = resampleStratified_withN(w,3000);
    theta_r = theta(inds,:);
    der_loglike = zeros(size(theta_r));
    der_logprior = zeros(size(theta_r));
    for i=1:size(theta_r,1)
        [~, der_logprior(i,:)] = der_logprior_fn(theta_r(i,:),options);
        [~, der_loglike(i,:)] = der_loglike_fn(theta_r(i,:),options);
    end
    KSD = KSD_fn(theta_r,der_loglike+der_logprior);

    % Cleaning and storing
    clearvars w theta_r inds der_loglike der_logprior;
    filename = sprintf('%s/ANS_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); save(filename);
    clearvars -except d k model_choice seed* options storage_location;


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% NS-SMC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Setting algorithm parameters
    filename = sprintf('%s/ANS_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); load(filename);
    options.N = 1000; options.levels = levels; options.cov_part = cov_part;

    % Running the algorithm
    clearvars -except d k model_choice seed* options storage_location; rng(seed_fixed);
    tic; [theta, log_weights, log_evidence, count_loglike, error_flag] = NS_SMC(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;

    % Calculating KSD
    w = (exp(log_weights - logsumexp(log_weights)));
    inds = resampleStratified_withN(w,3000);
    theta_r = theta(inds,:);
    der_loglike = zeros(size(theta_r));
    der_logprior = zeros(size(theta_r));
    for i=1:size(theta_r,1)
        [~, der_logprior(i,:)] = der_logprior_fn(theta_r(i,:),options);
        [~, der_loglike(i,:)] = der_loglike_fn(theta_r(i,:),options);
    end
    KSD = KSD_fn(theta_r,der_loglike+der_logprior);

    % Cleaning and storing
    clearvars w theta_r inds der_loglike der_logprior;
    filename = sprintf('%s/NS_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); save(filename);
    clearvars -except d k model_choice seed* storage_location;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% NS and NS*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Setting algorithm parameters
    options.N = 1000;
    options.stopping_epsilon = 10^(-8);
    options.model_choice = model_choice;
    options.h = 2.38/sqrt(d(model_choice));
    options.R = 10*model_choice;
    load('FA_data.mat'); options.y = data.y;

    % Running the algorithm
    clearvars -except d k model_choice seed* options storage_location; rng(seed_adapt);
    tic; [theta, log_weights, log_evidence, log_evidence_star, count_loglike, levels] = NS(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;

    % Calculating KSD (NS)
    w = (exp(log_weights - logsumexp(log_weights)));
    inds = resampleStratified_withN(w,3000);
    theta_r = theta(inds,:);
    der_loglike = zeros(size(theta_r));
    der_logprior = zeros(size(theta_r));
    for i=1:size(theta_r,1)
        [~, der_logprior(i,:)] = der_logprior_fn(theta_r(i,:),options);
        [~, der_loglike(i,:)] = der_loglike_fn(theta_r(i,:),options);
    end
    KSD_NS = KSD_fn(theta_r,der_loglike+der_logprior);

    % Calculating KSD (INS)
    t = 2:(length(levels)-1);
    log_weights_ns_star = levels(t) + log(((options.N-1)/options.N).^(t-1)) - log(options.N);
    w = (exp(log_weights_ns_star - logsumexp(log_weights_ns_star)));
    inds = resampleStratified_withN(w,3000);
    theta_r = theta(inds,:);
    der_loglike = zeros(size(theta_r));
    der_logprior = zeros(size(theta_r));
    for i=1:size(theta_r,1)
        [~, der_logprior(i,:)] = der_logprior_fn(theta_r(i,:),options);
        [~, der_loglike(i,:)] = der_loglike_fn(theta_r(i,:),options);
    end
    KSD_NS_star = KSD_fn(theta_r,der_loglike+der_logprior);

    % Cleaning and storing
    clearvars t log_weights_ns_star w theta_r inds der_loglike der_logprior;
    filename = sprintf('%s/NS_m%d_run%d.mat',storage_location,model_choice,seed_adapt); save(filename);
    clearvars -except d k model_choice seed* storage_location;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATA-SMC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Setting algorithm parameters
    options.N = 3000;
    options.alpha = exp(-1);
    options.model_choice = model_choice;
    options.h = 2.38/sqrt(d(model_choice));
    options.R = 10*model_choice;
    load('FA_data.mat'); options.y = data.y;

    % Running the algorithm
    clearvars -except d k model_choice seed* options storage_location; rng(seed_adapt);
    tic; [theta, log_evidence, count_loglike, gammavar, cov_part] = ATA_SMC(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;

    % Calculating KSD
    der_loglike = zeros(size(theta));
    der_logprior = zeros(size(theta));
    for i=1:size(theta,1)
        [~, der_logprior(i,:)] = der_logprior_fn(theta(i,:),options);
        [~, der_loglike(i,:)] = der_loglike_fn(theta(i,:),options);
    end
    KSD = KSD_fn(theta, der_loglike+der_logprior);

    % Cleaning and storing
    clearvars der_loglike der_logprior;
    filename = sprintf('%s/ATA_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); save(filename);
    clearvars -except d k model_choice seed* options storage_location;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%% TA-SMC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Setting algorithm parameters
    filename = sprintf('%s/ATA_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); load(filename);
    options.N = 3000; options.gammavar = gammavar; options.cov_part = cov_part;

    % Running the algorithm
    clearvars -except d k model_choice seed* options storage_location; rng(seed_fixed);
    tic; [theta, log_evidence, count_loglike] = TA_SMC(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;

    % Calculating KSD
    der_loglike = zeros(size(theta));
    der_logprior = zeros(size(theta));
    for i=1:size(theta,1)
        [~, der_logprior(i,:)] = der_logprior_fn(theta(i,:),options);
        [~, der_loglike(i,:)] = der_loglike_fn(theta(i,:),options);
    end
    KSD = KSD_fn(theta,der_loglike+der_logprior);

    % Cleaning and storing
    clearvars der_loglike der_logprior;
    filename = sprintf('%s/TA_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); save(filename);
    clearvars -except d k model_choice seed* storage_location;

    k
end
