% Turning the 1500 individual run files from 100 repeats * 5 methods * 3
% models into the single "results.mat" file that's needed for Figure 2 and
% Table 3.

count_ns = NaN(100,3);
time_ns = NaN(100,3);
log_evidence_ns = NaN(100,3);
log_evidence_ns_star = NaN(100,3);
ksd_ns = NaN(100,3);
ksd_ns_star = NaN(100,3);

count_ans_smc = NaN(100,3);
time_ans_smc = NaN(100,3);
log_evidence_ans_smc = NaN(100,3);
ksd_ans_smc = NaN(100,3);

count_ata_smc = NaN(100,3);
time_ata_smc = NaN(100,3);
log_evidence_ata_smc = NaN(100,3);
ksd_ata_smc = NaN(100,3);

count_ns_smc = NaN(100,3);
time_ns_smc = NaN(100,3);
log_evidence_ns_smc = NaN(100,3);
ksd_ns_smc = NaN(100,3);

count_ta_smc = NaN(100,3);
time_ta_smc = NaN(100,3);
log_evidence_ta_smc = NaN(100,3);
ksd_ta_smc = NaN(100,3);

for m = 1:3
    for rep = 1:100
        filename = sprintf("NS_m%d_run%d.mat",m,rep);
        load(filename);
        count_ns(rep,m) = count_loglike;
        time_ns(rep,m) = time;
        log_evidence_ns(rep,m) = log_evidence;
        log_evidence_ns_star(rep,m) = log_evidence_star;
        ksd_ns(rep,m) = KSD_NS;
        ksd_ns_star(rep,m) = KSD_NS_star;
        
        filename = sprintf("ANS_SMC_m%d_run%d.mat",m,rep);
        load(filename);
        count_ans_smc(rep,m) = count_loglike;
        time_ans_smc(rep,m) = time;
        log_evidence_ans_smc(rep,m) = log_evidence;
        ksd_ans_smc(rep,m) = KSD;
        
        filename = sprintf("ATA_SMC_m%d_run%d.mat",m,rep);
        load(filename);
        count_ata_smc(rep,m) = count_loglike;
        time_ata_smc(rep,m) = time;
        log_evidence_ata_smc(rep,m) = log_evidence;
        ksd_ata_smc(rep,m) = KSD;
        
        filename = sprintf("NS_SMC_m%d_run%d.mat",m,rep);
        load(filename);
        count_ns_smc(rep,m) = count_loglike + count_ans_smc(rep,m); % Because we did adaptive run first
        time_ns_smc(rep,m) = time + time_ans_smc(rep,m); % Because we did adaptive run first
        log_evidence_ns_smc(rep,m) = log_evidence;
        ksd_ns_smc(rep,m) = KSD;
        
        filename = sprintf("TA_SMC_m%d_run%d.mat",m,rep);
        load(filename);
        count_ta_smc(rep,m) = count_loglike + count_ata_smc(rep,m); % Because we did adaptive run first
        time_ta_smc(rep,m) = time + time_ata_smc(rep,m); % Because we did adaptive run first
        log_evidence_ta_smc(rep,m) = log_evidence;
        ksd_ta_smc(rep,m) = KSD;
    end
end

clearvars -except count_* time_* log_evidence_* ksd_*
clear log_evidence_star

save("results.mat");
