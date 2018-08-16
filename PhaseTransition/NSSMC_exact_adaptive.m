function [theta, log_weights, log_evidence, count_loglike, levels, distances] = NSSMC_exact_adaptive(options)
% Stratified SMC

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% options        -  options.N: Size of population of particles
%                -  options.rho: For choosing schedule
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
% levels         -  The thresholding levels

N = options.N;
rho = options.rho;
d = options.d;
stopping_number = options.stopping_number;
stopping_propZ = options.stopping_propZ;
stopping_ESS = options.stopping_ESS;

theta_curr = simprior_fn(N,options);
t = 1;

L_max=loglike_fn(options.c,options);

%initialise
log_evidence = -inf;

terminate = false;
theta = [];
log_weights = [];
levels = -inf;
distances = [1];
prod_c = 1;
m = 0;

prop_Z = 0;
ESS_early = 0;

loglike_curr = zeros(N,1);
for i=1:N
    loglike_curr(i) = loglike_fn(theta_curr(i,:),options);
end

last_dist = 1;
count_loglike = N;

while ~terminate
    t = t+1;
    ostats = sort(loglike_curr);
    levels(t) = ostats(floor((1-rho)*N));
    
    %---- decide whether or not to terminate -----
    if exp(levels(t) - L_max) > 0.75
        terminate = true;
        levels(t) = Inf; % forces final strata
    end
    %------------------------------------------
    
    el_ind = (loglike_curr>levels(t)); %indices of elite set
    
    num_elites = sum(el_ind);
    c = num_elites/N;
    
    % c = num_elites/N;
    if c == 1
        error('All Elites: Too many levels / Particles not moving')
    end
    if c == 0 && levels(t)~=inf
        error('No elites: Too many levels / Particles not moving')
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
    
    % New Samples
    if levels(t) ~= Inf
        last_dist = norm(theta_curr((loglike_curr == levels(t)),:));
        distances = [distances, last_dist];
        
        for i=1:N
            theta_curr(i,:) = Sampler(last_dist,d);
            loglike_curr(i,:) = loglike_fn(theta_curr(i,:),options);
        end
        count_loglike = count_loglike + N;
    end
    
end

%fprintf('Log evidence can also be calculated to be %.4f using the final weights.\n',logsumexp(log_weights)-log(N));
log_weights = log_weights - logsumexp(log_weights);

end
