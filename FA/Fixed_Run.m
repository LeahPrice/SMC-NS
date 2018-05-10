%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTING UP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_choice = 1;
rngno_fixed = 1;

p = fileparts(which('Fixed_Run.m')); % getting the location of the 'Fixed_Run.m' file
location = strcat(p(1:(strfind(p, '\FA'))), 'code'); % getting the location of the 'code' folder - Windows
% location = strcat(p(1:(strfind(p, '/FA'))), 'code'); % getting the location of the 'code' folder
addpath(genpath(location)); % adding the contents of the 'code' folder to the working directory

% Open parallel pool
% mypool = parcluster('local');
% mypool.JobStorageLocation=getenv('TMPDIR');
% parpool(mypool,8);

%%%%%%%%%%%%%%%%%%%%%%%%%% PERFORMING THE RUNS %%%%%%%%%%%%%%%%%%%%%%%%%%

% TA-SMC RW
filename = sprintf('SMC_TA_RW%d_tuning.mat',model_choice); load(filename);
options.N = 4000; options.gammavar = gammavar; options.R = R; options.cov_part = cov_part; options.h = h;

clearvars -except model_choice rngno_fixed options; rng(rngno_fixed);
tic; [theta, loglike, logprior, log_evidence, count_loglike] = SMC_TA_RW_fixed('loglike','logprior','simprior',options,true); time = toc; filename = sprintf('SMC_TA_RW%d_c%d.mat',model_choice,rngno_fixed); save(filename);

% TA-SMC MALA
filename = sprintf('SMC_TA_MALA%d_tuning.mat',model_choice); load(filename);
options.N = 4000; options.gammavar = gammavar; options.R = R; options.cov_part = cov_part; options.h = h;

clearvars -except model_choice rngno_fixed options; rng(rngno_fixed);
tic; [theta, loglike, logprior, der_loglike, der_logprior, log_evidence, count_loglike] = SMC_TA_MALA_fixed('der_loglike','der_logprior','simprior',options,true); time = toc; filename = sprintf('SMC_TA_MALA%d_c%d.mat',model_choice,rngno_fixed); save(filename);

% TA-SMC SLICE
filename = sprintf('SMC_TA_SLICE%d_tuning.mat',model_choice); load(filename);
options.N = 4000; options.gammavar = gammavar; options.std_pop = std_pop; options.R = R; options.w = w;

clearvars -except model_choice rngno_fixed options; rng(rngno_fixed);
tic; [theta, loglike, logprior, log_evidence, count_loglike] = SMC_TA_SLICE_fixed('loglike','logprior','simprior',options,true); time = toc; filename = sprintf('SMC_TA_SLICE%d_c%d.mat',model_choice,rngno_fixed); save(filename);

% NS-SMC RW
filename = sprintf('SMC_NS_RW%d_tuning.mat',model_choice); load(filename);
options.N = 1000; options.levels = levels; options.R = R; options.cov_part = cov_part; options.h = h;

clearvars -except model_choice rngno_fixed options; rng(rngno_fixed);
tic; [theta, log_weights, log_evidence, count_loglike] = SMC_NS_RW_fixed('loglike','logprior','simprior',options,true); time = toc; filename = sprintf('SMC_NS_RW%d_c%d.mat',model_choice,rngno_fixed); save(filename);

% NS-SMC MALA
filename = sprintf('SMC_NS_MALA%d_tuning.mat',model_choice); load(filename);
options.N = 1000; options.levels = levels; options.R = R; options.cov_part = cov_part; options.h = h;

clearvars -except model_choice rngno_fixed options; rng(rngno_fixed);
tic; [theta, log_weights, log_evidence, count_loglike] = SMC_NS_MALA_fixed('loglike','der_logprior','simprior',options,true); time = toc; filename = sprintf('SMC_NS_MALA%d_c%d.mat',model_choice,rngno_fixed); save(filename);

% NS-SMC SLICE
filename = sprintf('SMC_NS_SLICE%d_tuning.mat',model_choice);
load(filename);
options.N = 1000; options.levels = levels; options.std_pop = std_pop; options.R = R; options.w = w;

clearvars -except model_choice rngno_fixed options; rng(rngno_fixed);
tic; [theta, log_weights, log_evidence, count_loglike] = SMC_NS_SLICE_fixed('loglike','logprior','simprior',options,true); time = toc; filename = sprintf('SMC_NS_SLICE%d_c%d.mat',model_choice,rngno_fixed); save(filename);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLOSING DOWN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Close parallel pool
% delete(gcp);
