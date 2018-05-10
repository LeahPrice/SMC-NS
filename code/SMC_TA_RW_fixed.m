function [theta, loglike, logprior, log_evidence, count_loglike, gammavar, R, cov_part, h] = anneal_RW_fixed(loglike_fn,logprior_fn,simprior_fn,options,verbose)
% Likelihood annealing SMC with MH-MCMC RW move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function
%
% logprior_fn    -  log prior function
%
% simprior_fn    -  function to simulate from the prior
%
% options        -  options.N: Size of population of particles
%                -  options.gammavar: The fixed temperatures
%                -  options.R: fixed number of MCMC repeats
%                -  options.cov_part: fixed covariance matrices for RW
%                -  options.h: fixed multiplicative factors for RW
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
% log_evidence 	 -  The log evidence estimate (not necessarily unbiased when
%                   adapting the temperatures and proposals adaptively
%                   online)
%
% count_loglike  -  The total log likelihood computations

if isa(loglike_fn,'function_handle') == 0
    loglike_fn = str2func(loglike_fn);
end
if isa(logprior_fn,'function_handle') == 0
    logprior_fn = str2func(logprior_fn);
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

% initialise
log_evidence = 0;
T = length(gammavar);

loglike = zeros(N,1);
logprior = zeros(N,1);
parfor i=1:N
    loglike(i) = loglike_fn(theta(i,:),options);
    logprior(i) = logprior_fn(theta(i,:),options); % some of these will be wasted - maybe move to later
end

for t=1:T-1
    w = (gammavar(t+1)-gammavar(t))*loglike(:,t);
    log_evidence = log_evidence + logsumexp(w) - log(N);
    w = exp(w-logsumexp(w));
    
    % Sampling with replacement according to weights
    ind = resampleStratified(w);
    tplus1 = t + 1;
    theta(:,:,tplus1) = theta(ind,:,t);
    loglike(:,tplus1) = loglike(ind,t);
    logprior(:,tplus1) = logprior(ind,t);
    
    parfor i=1:N
        for k=1:R(t)
            theta_prop = mvnrnd(theta(i,:,tplus1),h(t)^2*cov_part{t});
            
            loglike_prop = loglike_fn(theta_prop,options);
            logprior_prop = logprior_fn(theta_prop,options);
            
            log_mh = gammavar(t+1)*(loglike_prop - loglike(i,tplus1)) + logprior_prop - logprior(i,tplus1);
            
            if exp(log_mh) > rand
                theta(i,:,tplus1) = theta_prop;
                loglike(i,tplus1) = loglike_prop;
                logprior(i,tplus1) = logprior_prop;
            end
        end
    end
end
count_loglike = N+N*sum(R);

end