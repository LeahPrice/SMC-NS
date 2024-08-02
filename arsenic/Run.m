% Organising file locations
p = fileparts(which('Run.m')); % getting the location of the 'Run.m' file
location = strcat(p(1:(strfind(p, '\arsenic'))), 'code'); % getting the location of the 'code' folder - Windows
% location = strcat(p(1:(strfind(p, '/arsenic'))), 'code'); % getting the location of the 'code' folder
storage_location = sprintf('%s/results',p);
addpath(genpath(location)); % adding the contents of the 'code' folder to the working directory

% Generating a matrix showing all possible models
% (a 1 indicates variable inclusion)
models = zeros(1,7);
for i=1:7
    models = [models; dec2bin(sum(nchoosek(2.^(0:7-1),i),2)) - '0'];
end
[num_models, ~] = size(models);

load('wells.mat')
[n, ~] = size(wells);

y = wells(:,1);

dist100 = wells(:,3)/100;
educ4 = wells(:,5)/4;
logarsenic = log(wells(:,2));
X = [ones(n,1) dist100 logarsenic educ4 ...
    dist100.*logarsenic dist100.*educ4 logarsenic.*educ4];

% Running for all model choices and repeats

for model_choice=2:num_models
    for seed_adapt=1:100
        seed_fixed = seed_adapt + 100;
        
        selected = find(models(model_choice,:));
        X_selected = X(:,selected);
        d = length(selected);
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ANS-SMC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Setting algorithm parameters
        options.N = 1000;
        options.alpha = exp(-1);
        options.stopping_epsilon = 10^(-5);
        options.h = 2.38/sqrt(d);
        options.R = 20;
        options.selected = selected; options.X = X_selected; options.d = d; options.y = y;
        
        % Running the algorithm
        clearvars -except d selected X_selected X y models model_choice seed* options storage_location; rng(seed_adapt);
        tic; [theta, log_weights, log_evidence, count_loglike, levels, cov_part, prod_c] = ANS_SMC(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;
        
        % Cleaning and storing
        clearvars w theta_r inds der_loglike der_logprior;
        filename = sprintf('%s/ANS_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); save(filename);
        clearvars -except d selected X_selected X y models model_choice seed* options storage_location;
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% NS-SMC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Setting algorithm parameters
        filename = sprintf('%s/ANS_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); load(filename);
        options.N = 1000; options.levels = levels; options.cov_part = cov_part;
        
        % Running the algorithm
        clearvars -except d selected X_selected X y models model_choice seed* options storage_location; rng(seed_fixed);
        tic; [theta, log_weights, log_evidence, count_loglike, error_flag] = NS_SMC(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;
        
        % Cleaning and storing
        clearvars w theta_r inds der_loglike der_logprior;
        filename = sprintf('%s/NS_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); save(filename);
        clearvars -except d selected X_selected X y models model_choice seed* storage_location;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% NS and NS*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Setting algorithm parameters
        options.N = 1000;
        options.stopping_epsilon = 10^(-8);
        options.model_choice = model_choice;
        options.h = 2.38/sqrt(d);
        options.R = 20;
        options.selected = selected; options.X = X_selected; options.d = d; options.y = y;
        
        % Running the algorithm
        clearvars -except d selected X_selected X y models model_choice seed* options storage_location; rng(seed_adapt);
        tic; [theta, log_weights, log_evidence, log_evidence_star, count_loglike, levels] = NS(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;
        
        % Cleaning and storing
        clearvars t log_weights_ns_star w theta_r inds der_loglike der_logprior;
        filename = sprintf('%s/NS_m%d_run%d.mat',storage_location,model_choice,seed_adapt); save(filename);
        clearvars -except d selected X_selected X y models model_choice seed* storage_location;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ATA-SMC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Setting algorithm parameters
        options.N = 3000;
        options.alpha = exp(-1);
        options.model_choice = model_choice;
        options.h = 2.38/sqrt(d);
        options.R = 20;
        options.selected = selected; options.X = X_selected; options.d = d; options.y = y;
        
        % Running the algorithm
        clearvars -except d selected X_selected X y models model_choice seed* options storage_location; rng(seed_adapt);
        tic; [theta, log_evidence, count_loglike, gammavar, cov_part] = ATA_SMC(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;
        
        % Cleaning and storing
        clearvars der_loglike der_logprior;
        filename = sprintf('%s/ATA_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); save(filename);
        clearvars -except d selected X_selected X y models model_choice seed* options storage_location;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%% TA-SMC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Setting algorithm parameters
        filename = sprintf('%s/ATA_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); load(filename);
        options.N = 3000; options.gammavar = gammavar; options.cov_part = cov_part;
        
        % Running the algorithm
        clearvars -except d selected X_selected X y models model_choice seed* options storage_location; rng(seed_fixed);
        tic; [theta, log_evidence, count_loglike] = TA_SMC(@loglike_fn,@logprior_fn,@simprior_fn,options,false); time = toc;
        
        % Cleaning and storing
        clearvars der_loglike der_logprior;
        filename = sprintf('%s/TA_SMC_m%d_run%d.mat',storage_location,model_choice,seed_adapt); save(filename);
        clearvars -except d selected X_selected X y models model_choice seed* storage_location;
        
        seed_adapt
    end
end
