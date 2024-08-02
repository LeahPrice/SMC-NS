% Turning the individual run files from 100 repeats * 5 methods * 127
% models into a single "results.mat" file.

p = fileparts(which('combine_results.m')); % getting the location of the 'Run.m' file
% location = strcat(p(1:(strfind(p, '\arsenic'))), 'code'); % getting the location of the 'code' folder - Windows
location = strcat(p(1:(strfind(p, '/arsenic'))), 'code'); % getting the location of the 'code' folder
addpath(genpath(location)); % adding the contents of the 'code' folder to the working directory

count_ns = NaN(100,128);
time_ns = NaN(100,128);
log_evidence_ns = NaN(100,128);
log_evidence_ns_star = NaN(100,128);

count_ans_smc = NaN(100,128);
time_ans_smc = NaN(100,128);
log_evidence_ans_smc = NaN(100,128);

count_ata_smc = NaN(100,128);
time_ata_smc = NaN(100,128);
log_evidence_ata_smc = NaN(100,128);

count_ns_smc = NaN(100,128);
time_ns_smc = NaN(100,128);
log_evidence_ns_smc = NaN(100,128);

count_ta_smc = NaN(100,128);
time_ta_smc = NaN(100,128);
log_evidence_ta_smc = NaN(100,128);

% Evidence for model 0
load('wells.mat')
[n, ~] = size(wells);
y = wells(:,1);
log_normcdf0 = log_normcdf(0);
log_evidence = sum(y*log_normcdf0 + (1-y)*log_normcdf0);

log_evidence_ns(:,1) = log_evidence;
log_evidence_ns_star(:,1) = log_evidence;
log_evidence_ans_smc(:,1) = log_evidence;
log_evidence_ata_smc(:,1) = log_evidence;
log_evidence_ns_smc(:,1) = log_evidence;
log_evidence_ta_smc(:,1) = log_evidence;

% Combining the results
for m = 2:128
    for rep = 1:100
        filename = sprintf("NS_m%d_run%d.mat",m,rep);
        load(filename);
        count_ns(rep,m) = count_loglike;
        time_ns(rep,m) = time;
        log_evidence_ns(rep,m) = log_evidence;
        log_evidence_ns_star(rep,m) = log_evidence_star;
        
        filename = sprintf("ANS_SMC_m%d_run%d.mat",m,rep);
        load(filename);
        count_ans_smc(rep,m) = count_loglike;
        time_ans_smc(rep,m) = time;
        log_evidence_ans_smc(rep,m) = log_evidence;
        
        filename = sprintf("ATA_SMC_m%d_run%d.mat",m,rep);
        load(filename);
        count_ata_smc(rep,m) = count_loglike;
        time_ata_smc(rep,m) = time;
        log_evidence_ata_smc(rep,m) = log_evidence;
        
        filename = sprintf("NS_SMC_m%d_run%d.mat",m,rep);
        load(filename);
        count_ns_smc(rep,m) = count_loglike + count_ans_smc(rep,m); % Because we did adaptive run first
        time_ns_smc(rep,m) = time + time_ans_smc(rep,m); % Because we did adaptive run first
        log_evidence_ns_smc(rep,m) = log_evidence;
        
        filename = sprintf("TA_SMC_m%d_run%d.mat",m,rep);
        load(filename);
        count_ta_smc(rep,m) = count_loglike + count_ata_smc(rep,m); % Because we did adaptive run first
        time_ta_smc(rep,m) = time + time_ata_smc(rep,m); % Because we did adaptive run first
        log_evidence_ta_smc(rep,m) = log_evidence;
    end
    m
end

clearvars -except count_* time_* log_evidence_*
clear log_evidence_star

save("results.mat");
