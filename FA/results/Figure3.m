% Run this for each model (1 to 3) and each method (uncomment below)
modeldim = 2;

load(sprintf('Gold_TA_SMC_m%d.mat',modeldim))
theta_gold = theta;
d = size(theta,2);

% Setting the dimension of interest
j = 14; % 14 corresponds to beta_{32}

% Getting the gold standard density ready
figure;
[vals_gold, x] = ksdensity(theta_gold(:,j));

% Overlaying with slightly transparent red lines for each of the 100 runs
for ii=1:100
    
    % ATA-SMC
    load(sprintf('ATA_SMC_m%d_run%d.mat',modeldim,ii))
    vals = ksdensity(theta(:,j),x); hold on;
    plot1 = plot(x,vals,'r','LineWidth',1.5);
    for i=1:length(plot1)
        plot1(i).Color = [plot1(i).Color 0.1];  % transparency alpha=0.1
    end
    
%     % NS
%     load(sprintf('NS_m%d_run%d.mat',modeldim,ii))
%     inds = resampleStratified(exp(log_weights-logsumexp(log_weights)));
%     theta = theta(inds,:);
%     vals = ksdensity(theta(:,j),x); hold on;
%     plot1 = plot(x,vals,'r','LineWidth',1.5);
%     for i=1:length(plot1)
%         plot1(i).Color = [plot1(i).Color 0.1];  % transparency alpha=0.1
%     end
%     
%     % ANS-SMC
%     load(sprintf('ANS_SMC_m%d_run%d.mat',modeldim,ii))
%     inds = resampleStratified(exp(log_weights-logsumexp(log_weights)));
%     theta = theta(inds,:);
%     vals = ksdensity(theta(:,j),x); hold on;
%     plot1 = plot(x,vals,'r','LineWidth',1.5);
%     for i=1:length(plot1)
%         plot1(i).Color = [plot1(i).Color 0.1];  % transparency alpha=0.1
%     end
end

% Increasing font size and adjusting x and y axes
set(gca,'FontSize',25);
xlim([min(x),max(x)]);
ylim([0,8]);

% Adding gold standard plot on top
plot(x,vals_gold,'color','k','LineWidth',4);
