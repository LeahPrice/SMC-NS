% Loading the runs
load("results.mat")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stable calculations of the model probabilities for each method
nc = logsumexp(log_evidence_ans_smc')'*ones(1,128);
modelprobs_ans_smc = exp(log_evidence_ans_smc - nc);

nc = logsumexp(log_evidence_ns_smc')'*ones(1,128);
modelprobs_ns_smc = exp(log_evidence_ns_smc - nc);

nc = logsumexp(log_evidence_ns')'*ones(1,128);
modelprobs_ns = exp(log_evidence_ns - nc);

nc = logsumexp(log_evidence_ata_smc')'*ones(1,128);
modelprobs_ata_smc = exp(log_evidence_ata_smc - nc);

nc = logsumexp(log_evidence_ta_smc')'*ones(1,128);
modelprobs_ta_smc = exp(log_evidence_ta_smc - nc);

modelprobs_94 = [modelprobs_ata_smc(:,94) modelprobs_ta_smc(:,94) modelprobs_ans_smc(:,94) modelprobs_ns_smc(:,94) modelprobs_ns(:,94)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting the model probabilities
h = boxplot(modelprobs_94);
ylabel('Estimated Model Probability');
label_all = {'ATA-SMC','TA-SMC','ANS-SMC','NS-SMC','NS'};
set(gca,'xticklabel',label_all,'xticklabelrotation',45);
set(gca,'FontSize',14.5);

% Changing the colour to black
set(h,'color','k');
h2 = findobj(gcf,'tag','Outliers');
for j=1:5
h2(j).MarkerEdgeColor = 'k';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Getting statistics about computation
mean([time_ata_smc(:,94) time_ta_smc(:,94) time_ans_smc(:,94) time_ns_smc(:,94) time_ns(:,94)])/60
mean([count_ata_smc(:,94) count_ta_smc(:,94) count_ans_smc(:,94) count_ns_smc(:,94) count_ns(:,94)])
