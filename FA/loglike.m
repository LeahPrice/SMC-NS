function [loglike] = loglike(x,options)

model_choice = options.model_choice;

n = length(options.y);

% Getting the location of the beta matrix diagonals (which have truncated
% normal prior
diags = 7;
temp = 7;
for j=1:model_choice-1
    temp = temp + 6-j+1;
    diags = [diags temp];
end

logged = [1:6 diags];

calc = x;
calc(logged) = exp(calc(logged));
beta = tril(ones(6,model_choice));
beta(beta==1) = calc(7:end);
covar = diag(calc(1:6)) + beta*beta';

loglike = -6*n/2*log(2*pi)-n/2*log(det(covar))-trace(options.y*inv(covar)*options.y')/2; %Just log of mvnpdf(y,0,covar)

end