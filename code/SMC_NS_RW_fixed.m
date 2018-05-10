function [theta, log_weights, log_evidence, count_loglike, error_flag] = stratified_RW_fixed(loglike_fn,logprior_fn,simprior_fn,options,verbose)
% Stratified SMC with MH-MCMC RW move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function
%
% logprior_fn    -  log prior function
%
% simprior_fn    -  function to simulate from the prior
%
% options        -  options.N: Size of population of particles
%                -  options.levels: The fixed levels
%                -  options.R: fixed number of MCMC repeats
%                -  options.cov_part: fixed covariance matrices for RW
%                -  options.h: fixed multiplicative factors for RW
%                -  ... example specific data and options
%
% verbose        -  set to true to get running update of progress and false otherwise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theta          -  Samples from the posterior (weights below)
%
% log_weights    -  Log weights for the samples
%
% log_evidence 	 -  The log evidence estimate (not necessarily unbiased when
%                   adapting the temperatures and proposals adaptively
%                   online)
%
% count_loglike  -  The total log likelihood computations
%
% error_flag     -  Zero if it finishes to level T, and 0 otherwise

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
levels = options.levels;
R = options.R;
cov_part = options.cov_part;
h = options.h;

theta_curr = simprior_fn(N,options);

%initialise
log_evidence = -inf;
T = length(levels);

theta = [];
log_weights = [];
prod_c = 1;
m = 0;
error_flag = NaN;

loglike_curr = zeros(N,1);
parfor i=1:N
    loglike_curr(i) = loglike_fn(theta_curr(i,:),options);
end

count_loglike = ones(N,1);

t = 2;
while t<=T
    el_ind = (loglike_curr>=levels(t)); %indices of elite set
    
    num_elites = sum(el_ind);
    c = num_elites/N;
    
    if c == 1
        error('All Elites: Too many levels / Particles not moving')
    end
    
    %--------- Early finish stats ---------
    log_weights_early = [log_weights; loglike_curr + log(prod_c)];
    log_evidence_early = logsumexp(log_weights_early) - log(N);
    ESS_early = exp(2*logsumexp(log_weights_early)-logsumexp(2*log_weights_early));
    
    %--- weighted (unnormalized) samples---
    logw = loglike_curr(~el_ind) + log(prod_c);
    term = logsumexp(logw) - log(N);
    log_evidence = logsumexp([log_evidence term]);
    %--------------------------------------
    
    %------ Storing the new samples ------
    new_inds = m+1 : m+sum(~el_ind);
    theta(new_inds,:) = theta_curr(~el_ind,:);
    log_weights(new_inds,1) = logw;
    m = max(new_inds);
    % ------------------------------------
    
    prop_Z = exp(log_evidence-log_evidence_early); % can tune to get this ~=1. Can also tune to get ESS_early/ESS_current~=1
    
    if verbose
        fprintf('\nIter %d\tLevel: %.4f\n\t\tEarly ESS: %.4f\n\t\tEarly log Z: %.4f\n\t\tProportion of Early Z: %.4f\n\t\tTotal elite: %d\n',t,levels(t),ESS_early,log_evidence_early,prop_Z,num_elites);
    end
    
    prod_c = prod_c*c;
        
    if c == 0 && t~=T
        error_flag = t;
        t = T;
    end
    
    % Resample and Move
    % Note --- all the weights are equal so we just sample uniformly
    if levels(t) ~= Inf
        resamp_weights = zeros(N,1);
        resamp_weights(el_ind) = ones(num_elites,1)/num_elites;
        inds = resampleStratified(resamp_weights);
        
        if t==2 %need to evaluate log prior for the first time
            logprior_curr = zeros(N,1);
            for i=unique(inds)
                logprior_curr(i) = logprior_fn(theta_curr(i,:),options);
            end
        end
        
        theta_curr = theta_curr(inds,:);
        loglike_curr = loglike_curr(inds);
        logprior_curr = logprior_curr(inds);
        
        parfor i=1:N
            for k=1:R(t)
                theta_prop = mvnrnd(theta_curr(i,:),h(t)^2*cov_part{t});
                logprior_prop = logprior_fn(theta_prop,options);
                
                if exp(logprior_prop - logprior_curr(i)) > rand
                    loglike_prop = loglike_fn(theta_prop,options);
                    count_loglike(i) = count_loglike(i) + 1;
                    
                    if loglike_prop>=levels(t)
                        theta_curr(i,:) = theta_prop;
                        loglike_curr(i) = loglike_prop;
                        logprior_curr(i) = logprior_prop;
                    end
                end
            end
        end
    end
	t = t + 1;
    
end

%fprintf('Log evidence can also be calculated to be %.4f using the final weights.\n',logsumexp(log_weights)-log(N));
log_weights = log_weights - logsumexp(log_weights);

count_loglike = sum(count_loglike);

end