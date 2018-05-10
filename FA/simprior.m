function samples = simprior(N,options)

model_choice = options.model_choice;

%Draws from prior
C0 = 1;
IG1 = 2.2/2;
IG2 = (0.1/2)^(-1);

% Inverse gamma priors for diagonals of Sigma and standard normal priors for the beta (except for diagonals which are truncated standard normal)
samples = [-1*log(gamrnd(IG1,IG2,N,6)) normrnd(0,C0,[N,6*model_choice-sum(1:(model_choice-1))])];

%Locations of beta matrix diagonals
diags = 7;
temp = 7;
for j=1:model_choice-1
    temp = temp + 6-j+1;
    diags = [diags temp];
end

%Truncating the normal prior for beta diagonals and taking log
samples(:,diags) = log(abs(samples(:,diags)));

end