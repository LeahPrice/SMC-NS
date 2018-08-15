function [theta, log_weights, log_evidence, count_loglike, error_flag] = stratified_MCMC_fixed(options)
% Stratified SMC

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% options        -  options.N: Size of population of particles
%                -  options.levels: The fixed levels
%                -  options.R: fixed number of MCMC repeats
%                -  options.sig: fixed parameter for sampler
%                -  options.d: dimension

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

N = options.N;
levels = options.levels;
R = options.R;
d = options.d;
sig = options.sig;

theta_curr = simprior_fn(N,options);

%initialise
log_evidence = -inf;
T = length(levels);

theta = [];
log_weights = [];
prod_c = 1;
m = 0;
error_flag = 0;

loglike_curr = zeros(N,1);
for i=1:N
    loglike_curr(i) = loglike_fn(theta_curr(i,:),options);
end

count_loglike = ones(N,1);

for t=2:T
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
        
    prod_c = prod_c*c;

    if c == 0 && levels(t) ~= inf
        error_flag = 1;
        warning('Particle System has died at level %.0f of %.0f.', t, length(levels))
        t = T;
    end

    % Resample and Move
    % Note --- all the weights are equal so we just sample uniformly
    if levels(t) ~= Inf
        resamp_weights = zeros(N,1);
        resamp_weights(el_ind) = ones(num_elites,1)/num_elites;
        %inds = resampleStratified(resamp_weights);
        inds = resampleMultinomial(resamp_weights);
        
        theta_curr = theta_curr(inds,:);
        loglike_curr = loglike_curr(inds);
        
        for i=1:N
            for k=1:R
                if rand < 0.5
                    sig = options.sig;
                else
                    sig = options.sig/4;
                end
                theta_prop = theta_curr(i,:);
                loc = ceil(rand*d);
                theta_prop(loc) = theta_prop(loc) + sig*randn;
                if norm(theta_prop)<1
                    loglike_prop = loglike_fn(theta_prop,options);
                    count_loglike(i) = count_loglike(i) + 1;
                    if loglike_prop > levels(t)
                        theta_curr(i,:) = theta_prop;
                        loglike_curr(i,:) = loglike_prop;
                    end
                end
            end
        end
    end
    
end

%fprintf('Log evidence can also be calculated to be %.4f using the final weights.\n',logsumexp(log_weights)-log(N));
log_weights = log_weights - logsumexp(log_weights);

count_loglike = sum(count_loglike);

end