function [theta, loglike, logprior, der_loglike, der_logprior, log_evidence, count_loglike] = anneal_MALA_fixed(der_loglike_fn,der_logprior_fn,simprior_fn,options,verbose)
% Likelihood annealing SMC with MH-MCMC MALA move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% der_loglike_fn - derivative of the log likelihood function
%
% der_logprior_fn- derivative of the log prior function
%
% simprior_fn    -  function to simulate from the prior
%
% options        -  options.N: Size of population of particles
%                -  options.gammavar: The fixed temperatures
%                -  options.R: fixed number of MCMC repeats
%                -  options.cov_part: fixed covariance matrices for MALA
%                -  options.h: fixed multiplicative factors for MALA
%                -  ... example specific data and options
%
% verbose        -  set to true to get running update of progress and false otherwise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theta          -  Final N samples from each temperature
%
% loglike        -  Log likelihood corresponding to above thetas.
%
% logprior       -  Log prior corresponding to above thetas.
%
% der_loglike    -  Derivative of the log likelihood corresponding to above thetas.
%
% der_logprior   -  Derivative of the log prior corresponding to above thetas.
%
% log_evidence 	 -  The log evidence estimate (not necessarily unbiased when
%                   adapting the temperatures and proposals adaptively
%                   online)
%
% count_loglike  -  The total log likelihood computations

if isa(der_loglike_fn,'function_handle') == 0
    der_loglike_fn = str2func(der_loglike_fn);
end
if isa(der_logprior_fn,'function_handle') == 0
    der_logprior_fn = str2func(der_logprior_fn);
end
if isa(simprior_fn,'function_handle') == 0
    simprior_fn = str2func(simprior_fn);
end

warning('off','all');

N = options.N;
gammavar = options.gammavar;
R = options.R;
cov_part = options.cov_part;
h = options.h;

theta = simprior_fn(N,options);
d = size(theta,2);

% initialise
log_evidence = 0;
T = length(gammavar);

loglike = zeros(N,1); der_loglike = zeros(N,d);
logprior = zeros(N,1); der_logprior = zeros(N,d);
parfor i=1:N
    [loglike(i), der_loglike(i,:)] = der_loglike_fn(theta(i,:),options); % some of the derivative calculations will be wasted...
    [logprior(i), der_logprior(i,:)] = der_logprior_fn(theta(i,:),options); % some of these will be wasted - maybe move to later
end

for t=1:T-1
    w = (gammavar(t+1)-gammavar(t))*loglike(:,t);
    log_evidence = log_evidence + logsumexp(w) - log(N);
    w = exp(w-logsumexp(w));
    
    % Sampling with replacement according to weights
    resamp_ind = resampleStratified(w);
    tplus1 = t + 1;
    theta(:,:,tplus1) = theta(resamp_ind,:,t);
    loglike(:,tplus1) = loglike(resamp_ind,t);
    der_loglike(:,:,tplus1) = der_loglike(resamp_ind,:,t);
    logprior(:,tplus1) = logprior(resamp_ind,t);
    der_logprior(:,:,tplus1) = der_logprior(resamp_ind,:,t);
        
    % Performing remaining repeats
    parfor i=1:N
        for k=1:R(t)
            %MALA
            mymean = theta(i,:,tplus1)' + h(t)^2/2*cov_part{t}*(gammavar(t+1)*der_loglike(i,:,tplus1)'+der_logprior(i,:,tplus1)');
            theta_prop = mvnrnd(mymean,h(t)^2*cov_part{t});
            
            [loglike_prop, der_loglike_prop] = der_loglike_fn(theta_prop,options);
            [logprior_prop, der_logprior_prop] = der_logprior_fn(theta_prop,options);
            
            mymean_prop = theta_prop' + h(t)^2/2*cov_part{t}*(gammavar(t+1)*der_loglike_prop'+der_logprior_prop');
            transition_totheta = log_mvnpdf(theta(i,:,tplus1)',mymean_prop,h(t)^2*cov_part{t});
            transition_toprop = log_mvnpdf(theta_prop',mymean,h(t)^2*cov_part{t});
            
            log_mh = gammavar(t+1)*loglike_prop - gammavar(t+1)*loglike(i,tplus1) + logprior_prop - logprior(i,tplus1) + transition_totheta - transition_toprop;
            
            if exp(log_mh) > rand
                theta(i,:,tplus1) = theta_prop;
                loglike(i,tplus1) = loglike_prop;
                der_loglike(i,:,tplus1) = der_loglike_prop;
                logprior(i,tplus1) = logprior_prop;
                der_logprior(i,:,tplus1) = der_logprior_prop;
            end
        end
    end
end
count_loglike = N+N*sum(R);

end