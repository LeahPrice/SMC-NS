function logprior = logprior(x,options)

model_choice = options.model_choice;

% Dimension of theta
p = length(x);

% Parameters for prior
C0 = 1;
IG1 = 2.2/2;
IG2 = (0.1/2)^(-1);

logprior = zeros(1,p);
for j=1:6
    logprior(j) = x(j) - 1/IG2./exp(x(j))-log(gamma(IG1))-IG1*log(IG2)-x(j)*(IG1+1); %Inverse gamma with log taken.
end

if model_choice>0
    % Getting the location of the beta matrix diagonals (which have truncated
    % normal prior
    diags = 7;
    temp = 7;
    for j=1:model_choice-1
        temp = temp + 6-j+1;
        diags = [diags temp];
    end
    
    % Getting the off-diagonal locations (standard normal prior)
    temp2 = 7:p;
    offdiag = temp2(ismember(temp2,diags)==0);
    
    for j=diags
        logprior(j) = x(j) + log_normpdf(exp(x(j)),0,C0)-log(1-normcdf(0,0,C0)); %Truncated normal with log taken
    end
    for j=offdiag
        logprior(j) = log_normpdf(x(j),0,C0); %Normal with no logs
    end
end

logprior = sum(logprior);

end