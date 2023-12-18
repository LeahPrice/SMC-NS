function [loglike, der_loglike] = der_loglike_fn(x,options)

model_choice = options.model_choice;

p = length(x); % number of parameters
n = length(options.y);
num_vars = 6;
der_loglike = zeros(1,p);

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

%dSigma/dthetaj for j=1:6 is a matrix of zeros with (j,j) component = 1. Then want dSigma/dtransf_thetaj so multiply by exp(theta(j));
for j=1:6
    dcovar_dtransf_theta{j} = zeros(num_vars,num_vars);
    dcovar_dtransf_theta{j}(j,j) = exp(x(j));
end

for i=1:num_vars
    for j=1:model_choice
        myindex = find(calc==beta(i,j));
        if isempty(myindex)==0
            dcovar_dtransf_theta{myindex} = zeros(num_vars,num_vars);
            dcovar_dtransf_theta{myindex}(i,:) = beta(:,j);
            dcovar_dtransf_theta{myindex}(:,i) = beta(:,j)';
            dcovar_dtransf_theta{myindex}(i,i) = 2*beta(i,j);
            if ismember(myindex,diags)
                dcovar_dtransf_theta{myindex} = dcovar_dtransf_theta{myindex}*exp(x(myindex));
            end
        end
    end
end

inv_covar = inv(covar);
for j=1:p
    der_loglike(j) = -n*0.5*trace(inv_covar*dcovar_dtransf_theta{j}) + trace(0.5*options.y*inv_covar*dcovar_dtransf_theta{j}*inv_covar*options.y');
end

end