% Do all combinations of N (100/1000/10000) and sampler (exact/RW)
% Uncomment ata_smc and ta_smc for RW runs
load('SpikeSlab_exact_N100.mat')

% Getting the gold standard
e1 = 0.1 * chi2cdf(0.1^(-2), d, 'upper');
e2 = 0.9 * chi2cdf(0.01^(-2), d, 'upper');
eps = e1 + e2;
logV = (d/2)*log(pi) - gammaln(d/2 + 1);
logGold = log(1-eps) - logV;
gold = exp(logGold);

% Number of tests we're performing in the table
num_bonferroni = 30;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% For exact runs

% Average evidence
mean(exp([log_evidence_ns log_evidence_ns_star log_evidence_ans_smc...
    log_evidence_ns_smc]))

% Standard error
std(exp([log_evidence_ns log_evidence_ns_star log_evidence_ans_smc...
    log_evidence_ns_smc]))/sqrt(num_repeats)

% Compute time
mean([time_ns time_ns_star time_ans_smc...
time_ns_smc])

% Average log likelihood evaluations
mean([count_ns count_ns_star count_ans_smc count_ns_smc])

% Hypothesis tests for the evidence (for red colouring)
[ttest(exp(log_evidence_ns),gold,'Alpha',0.05/num_bonferroni)...
    ttest(exp(log_evidence_ns_star),gold,'Alpha',0.05/num_bonferroni)...
    ttest(exp(log_evidence_ans_smc),gold,'Alpha',0.05/num_bonferroni)...
    ttest(exp(log_evidence_ns_smc),gold,'Alpha',0.05/num_bonferroni)]

% MSE (for bolding)
[mean((exp(log_evidence_ns) - gold).^2)...
mean((exp(log_evidence_ns_star) - gold).^2)...
mean((exp(log_evidence_ans_smc) - gold).^2)...
mean((exp(log_evidence_ns_smc) - gold).^2)]

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % For RW
% 
% % Average evidence
% mean(exp([log_evidence_ns log_evidence_ns_star log_evidence_ans_smc...
%     log_evidence_ns_smc log_evidence_ata_smc log_evidence_ta_smc]))
% 
% % Standard error
% std(exp([log_evidence_ns log_evidence_ns_star log_evidence_ans_smc...
%     log_evidence_ns_smc log_evidence_ata_smc log_evidence_ta_smc]))/sqrt(num_repeats)
% 
% % Compute time
% mean([time_ns time_ns_star time_ans_smc...
%     time_ns_smc time_ata_smc time_ta_smc])
% 
% % Average log likelihood evaluations
% mean([count_ns count_ns_star count_ans_smc count_ns_smc...
%     count_ata_smc count_ta_smc])
% 
% % Hypothesis tests for the evidence (for red colouring)
% [ttest(exp(log_evidence_ns),gold,'Alpha',0.05/num_bonferroni)...
%     ttest(exp(log_evidence_ns_star),gold,'Alpha',0.05/num_bonferroni)...
%     ttest(exp(log_evidence_ans_smc),gold,'Alpha',0.05/num_bonferroni)...
%     ttest(exp(log_evidence_ns_smc),gold,'Alpha',0.05/num_bonferroni)...
%     ttest(exp(log_evidence_ata_smc),gold,'Alpha',0.05/num_bonferroni)...
%     ttest(exp(log_evidence_ta_smc),gold,'Alpha',0.05/num_bonferroni)]
% 
% % MSE (for bolding)
% [mean((exp(log_evidence_ns) - gold).^2)...
% mean((exp(log_evidence_ns_star) - gold).^2)...
% mean((exp(log_evidence_ans_smc) - gold).^2)...
% mean((exp(log_evidence_ns_smc) - gold).^2)...
% mean((exp(log_evidence_ata_smc) - gold).^2)...
% mean((exp(log_evidence_ta_smc) - gold).^2)]


