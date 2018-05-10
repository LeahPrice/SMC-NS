%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTING UP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

rngno_fixed = 1; % set random number generator for fixed runs

p = fileparts(which('Fixed_Run.m')); % getting the location of the 'Run.m' file
location = strcat(p(1:(strfind(p, '\ODE'))), 'code'); % getting the location of the 'code' folder - Windows
% location = strcat(p(1:(strfind(p, '/ODE'))), 'code'); % getting the location of the 'code' folder
addpath(genpath(location)); % adding the contents of the 'code' folder to the working directory

% Open parallel pool
% mypool = parcluster('local');
% mypool.JobStorageLocation=getenv('TMPDIR');
% parpool(mypool,8);

%%%%%%%%%%%%%%%%%%%%%%%%%% PERFORMING THE RUNS %%%%%%%%%%%%%%%%%%%%%%%%%%

% TA-SMC RW
load('SMC_TA_RW_tuning.mat');
options.N = 4000; options.gammavar = gammavar; options.R = R; options.cov_part = cov_part; options.h = h;

clearvars -except rngno_fixed options; rng(rngno_fixed);
tic; [theta, loglike, logprior, log_evidence, count_loglike] = SMC_TA_RW_fixed('loglike','logprior','simprior',options,true); time = toc; filename = sprintf('SMC_TA_RW_c%d.mat',rngno_fixed); save(filename);

% TA-SMC MALA
load('SMC_TA_MALA_tuning.mat');
options.N = 4000; options.gammavar = gammavar; options.R = R; options.cov_part = cov_part; options.h = h;

clearvars -except rngno_fixed options; rng(rngno_fixed);
tic; [theta, loglike, logprior, der_loglike, der_logprior, log_evidence, count_loglike] = SMC_TA_MALA_fixed('der_loglike','der_logprior','simprior',options,true); time = toc; filename = sprintf('SMC_TA_MALA_c%d.mat',rngno_fixed); save(filename);

% TA-SMC SLICE
load('SMC_TA_SLICE_tuning.mat');
options.N = 4000; options.gammavar = gammavar; options.std_pop = std_pop; options.R = R; options.w = w;

clearvars -except rngno_fixed options; rng(rngno_fixed);
tic; [theta, loglike, logprior, log_evidence, count_loglike] = SMC_TA_SLICE_fixed('loglike','logprior','simprior',options,true); time = toc; filename = sprintf('SMC_TA_SLICE_c%d.mat',rngno_fixed); save(filename);

% NS-SMC RW
load('SMC_NS_RW_tuning.mat');
options.N = 1000; options.levels = levels; options.R = R; options.cov_part = cov_part; options.h = h;

clearvars -except rngno_fixed options; rng(rngno_fixed);
tic; [theta, log_weights, log_evidence, count_loglike, error_flag] = SMC_NS_RW_fixed('loglike','logprior','simprior',options,true); time = toc; filename = sprintf('SMC_NS_RW_c%d.mat',rngno_fixed); save(filename);

% NS-SMC MALA
load('SMC_NS_MALA_tuning.mat');
options.N = 1000; options.levels = levels; options.R = R; options.cov_part = cov_part; options.h = h;

clearvars -except rngno_fixed options; rng(rngno_fixed);
tic; [theta, log_weights, log_evidence, count_loglike, error_flag] = SMC_NS_MALA_fixed('loglike','der_logprior','simprior',options,true); time = toc; filename = sprintf('SMC_NS_MALA_c%d.mat',rngno_fixed); save(filename);

% NS-SMC SLICE
load('SMC_NS_SLICE_tuning.mat');
options.N = 1000; options.levels = levels; options.std_pop = std_pop; options.R = R; options.w = w;

clearvars -except rngno_fixed options; rng(rngno_fixed);
tic; [theta, log_weights, log_evidence, count_loglike, error_flag] = SMC_NS_SLICE_fixed('loglike','logprior','simprior',options,true); time = toc; filename = sprintf('SMC_NS_SLICE_c%d.mat',rngno_fixed); save(filename);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLOSING DOWN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Close parallel pool
% delete(gcp);
