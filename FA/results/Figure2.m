%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loading the results

% Loading the gold standard
log_evidence_gold = NaN(3,1);
load("Gold_TA_SMC_m1.mat");
log_evidence_gold(1) = log_evidence;
load("Gold_TA_SMC_m2.mat");
log_evidence_gold(2) = log_evidence;
load("Gold_TA_SMC_m3.mat");
log_evidence_gold(3) = log_evidence;
clearvars -except log_evidence_gold

% Loading the runs
load("results.mat")

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Stable calculations of the model probabilities for each method
mymax = max(log_evidence_gold);
logsumexp_gold = mymax + log(sum(exp(log_evidence_gold - mymax)));
modelprobs_gold = exp(log_evidence_gold - logsumexp_gold*ones(1,3));

mymax = max(log_evidence_ata_smc,[],2);
nc = mymax + log(sum(exp(log_evidence_ata_smc - mymax),2));
modelprobs_ata_smc = exp(log_evidence_ata_smc - nc*ones(1,3));

mymax = max(log_evidence_ans_smc,[],2);
nc = mymax + log(sum(exp(log_evidence_ans_smc - mymax),2));
modelprobs_ans_smc = exp(log_evidence_ans_smc - nc*ones(1,3));

mymax = max(log_evidence_ns_smc,[],2);
nc = mymax + log(sum(exp(log_evidence_ns_smc - mymax),2));
modelprobs_ns_smc = exp(log_evidence_ns_smc - nc*ones(1,3));

mymax = max(log_evidence_ns,[],2);
nc = mymax + log(sum(exp(log_evidence_ns - mymax),2));
modelprobs_ns = exp(log_evidence_ns - nc*ones(1,3));

modelprobs_1 = [modelprobs_ata_smc(:,1) modelprobs_ans_smc(:,1) modelprobs_ns_smc(:,1) modelprobs_ns(:,1)];
modelprobs_2 = [modelprobs_ata_smc(:,2) modelprobs_ans_smc(:,2) modelprobs_ns_smc(:,2) modelprobs_ns(:,2)];
modelprobs_3 = [modelprobs_ata_smc(:,3) modelprobs_ans_smc(:,3) modelprobs_ns_smc(:,3) modelprobs_ns(:,3)];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotting the model probabilities
h = boxplot([modelprobs_1 modelprobs_2 modelprobs_3],'positions',[0.75 1.5 2.25 3  4.5 5.25 6 6.75  8.25 9 9.75 10.5]);
xlim([0, 11.25]);
ylabel('Estimated Model Probability');
label_all = {'ATA-SMC','ANS-SMC','NS-SMC','NS','ATA-SMC','ANS-SMC','NS-SMC','NS','ATA-SMC','ANS-SMC','NS-SMC','NS'};
set(gca,'xticklabel',label_all,'xticklabelrotation',45);
set(gca,'FontSize',14.5);
xlabel(blanks(15)+"1 Factor"+blanks(27)+"2 Factors"+blanks(27)+"3 Factors"+blanks(15))
line([0.5,3.25],[modelprobs_gold(1),modelprobs_gold(1)],'Color','k')
line([4.25,7],[modelprobs_gold(2),modelprobs_gold(2)],'Color','k')
line([8,10.75],[modelprobs_gold(3),modelprobs_gold(3)],'Color','k')

% Changing the colour to black
set(h,'color','k');
h2 = findobj(gcf,'tag','Outliers');
for j=1:21
h2(j).MarkerEdgeColor = 'k';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Zoomed-in plot of the model one probabilities
figure; 
h = boxplot(modelprobs_1,'positions',[0.5 1.25 2 2.75]);
line([0.25,3],[modelprobs_gold(1),modelprobs_gold(1)],'Color','k')
set(gca,'xticklabel',{});
set(gca,'FontSize',15);

% Changing the colour to black
set(h,'color','k');
h2 = findobj(gcf,'tag','Outliers');
for j=1:7
    h2(j).MarkerEdgeColor = 'k';
end
