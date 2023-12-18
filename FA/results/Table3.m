load("results.mat");

% Results for Table 3
for m=1:3
    mean([ksd_ata_smc(:,m) ksd_ans_smc(:,m) ksd_ns_smc(:,m) ksd_ns(:,m)])
    mean([count_ata_smc(:,m) count_ans_smc(:,m) count_ns_smc(:,m) count_ns(:,m)])
end
