function [theta, log_weights, log_evidence, log_evidence_star, count_loglike, levels] = NS(loglike_fn,logprior_fn,simprior_fn,options,verbose)
% NS with MH-MCMC RW move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function taking the samples and a list as input
%
% logprior_fn    -  log prior function taking the samples and a list as input
%
% simprior_fn    -  function to simulate from the prior taking the sample size and list as input
%
% options        -  options.N: Size of population of particles
%                -  options.stopping_epsilon: epsilon for NS stopping rule
%                -  options.h: Multiplicative factor for the covariance is h^2
%                -  options.R: Number of MCMC repeats per particle per SMC iteration
%                -  ... example specific data and options
%
% verbose        -  set to true to get running update of progress and false otherwise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theta          -  Samples from the posterior (weights below)
%
% log_weights    -  Log weights for the samples
%
% log_evidence 	 -  The log evidence estimate from vanilla NS
%
% log_evidence_star 	 -  The log evidence estimate from NS*
%
% count_loglike  -  The total log likelihood computations
%
% levels         -  The levels for nested sampling

if isa(loglike_fn,'function_handle') == 0
    loglike_fn = str2func(loglike_fn);
end
if isa(logprior_fn,'function_handle') == 0
    logprior_fn = str2func(logprior_fn);
end
if isa(simprior_fn,'function_handle') == 0
    simprior_fn = str2func(simprior_fn);
end

N = options.N;
stopping_epsilon = options.stopping_epsilon; % e.g. 10^(-8)

theta_curr = simprior_fn(N,options);

%initialise
log_evidence = -inf;
log_evidence_star = -inf;
t = 1;
h = options.h;
R = options.R;

theta = [];
loglikes = [];
log_weights = [];
levels = -inf;

loglike_curr = zeros(N,1);
logprior_curr = zeros(N,1);
for i=1:N
    loglike_curr(i) = loglike_fn(theta_curr(i,:),options);
    logprior_curr(i) = logprior_fn(theta_curr(i,:),options);
end
count_loglike = N;

while -t/N + max(loglike_curr) > log(stopping_epsilon)  + log_evidence
% while count_loglike <= options.desired_count
    t = t+1;
    [levels(t), min_loc] = min(loglike_curr);
    
    log_weight = levels(t) + log(exp(-(t-1)/N)-exp(-t/N)); % Riemann
    %log_weight = levels(t) + log((exp(-(t-1)/N)-exp(-(t+1)/N))/2); % trapezoidal
    
    % Storing the new samples (COULD CONSIDER DOING THIS MORE EFFICIENTLY!)
    theta(t-1,:) = theta_curr(min_loc,:);
    loglikes(t-1) = loglike_curr(min_loc);
    log_weights(t-1) = log_weight;
    % ------------------------------------------------
    
    log_evidence = logsumexp([log_evidence log_weight]);
    
    % adjusted version
    log_weight_star = levels(t) + log(((N-1)/N)^(t-1)) - log(N);
    log_evidence_star = logsumexp([log_evidence_star log_weight_star]);

    if verbose
        fprintf('\nIter %d\tLevel: %.4f\n\t\tCurrent log Z: %.4f\n',t,levels(t),log_evidence);
    end
    
    choice=min_loc;
    while choice==min_loc
        choice=ceil(rand*N);
    end
    theta_curr(min_loc,:) = theta_curr(choice,:);
    loglike_curr(min_loc) = loglike_curr(choice);
    logprior_curr(min_loc) = logprior_curr(choice);
    
    cov_rw = cov(theta_curr);
    
    for k = 1:R
        theta_prop = mvnrnd(theta_curr(min_loc,:),h^2*cov_rw);
        
        logprior_prop = logprior_fn(theta_prop,options);
        MH = exp(logprior_prop - logprior_curr(min_loc));
        if MH > rand
            loglike_prop = loglike_fn(theta_prop,options);
            count_loglike = count_loglike + 1;
            if loglike_prop>levels(t)
                theta_curr(min_loc,:) = theta_prop;
                loglike_curr(min_loc) = loglike_prop;
                logprior_curr(min_loc) = logprior_prop;
            end
        end
    end
    
end

t = t+1;
levels(t) = Inf; % forces final strata
log_evidence = logsumexp([log_evidence logsumexp(loglike_curr - t/N) - log(N)]);

prob = t*log((N-1)/N);

log_evidence_star = logsumexp([log_evidence_star, logsumexp(loglike_curr + prob - log(N))]);

end