function [theta, log_weights, log_evidence, count_loglike, log_evidence_adj] = nested_exact(options)
% Nested sampling

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% options        -  options.N: Size of population of particles
%                -  options.tol: The tolerance levels
%                -  options.R: fixed number of MCMC repeats
%                -  options.sig: fixed parameter for sampler
%                -  options.d: dimension

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theta          -  Samples from the posterior (weights below)
%
% log_weights    -  Log weights for the samples
%
% log_evidence   -  The log evidence estimate (not necessarily unbiased when
%                   adapting the temperatures and proposals adaptively
%                   online)
%
% count_loglike  -  The total log likelihood computations required for the
%                   method.
%
% levels         -  The levels for nested sampling
%
% cov_part       -  The covariance of the particles used in the move step (saved
%                   at every 25 iterations)

N = options.N;

theta = []; log_weights = []; % for inference
theta_curr = simprior_fn(N,options);

%initialise
log_evidence = -inf;
log_evidence_adj = -inf;
L_t = -inf;
t = 0;

c = options.c;
L_max=loglike_fn(c,options);

loglike_curr = zeros(N,1);

for i=1:N
    loglike_curr(i) = loglike_fn(theta_curr(i,:),options);
end
count_loglike = N;

while exp(L_t-L_max) < 0.75
    t = t+1;
    
    [L_t, min_loc] = min(loglike_curr);
    %log_weight = L_t + log(exp(-(t-1)/N)-exp(-t/N)); % Riemann sum
    log_weight = L_t + log(exp(-(t-1)/N)-exp(-(t+1)/N)) - log(2); %Trapezoidal Rule
    % fprintf('Current val: %.2e , max_val: %.2e, ratio: %.2e \n', L_t, L_max, exp(L_t-L_max))
    % log_evidence = logsumexp([log_evidence log_weight]);
   log_evidence = logplusexp(log_evidence,log_weight);
     
   % adjusted version
   log_weight_adj = L_t + log(((N-1)/N)^(t-1)) - log(N);
   %log_evidence_adj = logsumexp([log_evidence_adj log_weight_adj]);
   log_evidence_adj = logplusexp(log_evidence_adj, log_weight_adj);
      
   %------ Storing the new samples and weights ------
   % theta(t,:) = theta_curr(min_loc,:);
   % log_weights(t) = log_weight;
   % ------------------------------------------------
    
    %log_evidence_early = logsumexp([log_evidence logsumexp(loglike_curr - (t-1)/N) - log(N)]);
    %ESS = exp(-logsumexp(2*(log_weights-log_evidence)));
    
    % Sample
    last_dist = norm(theta_curr(min_loc,:));
    theta_curr(min_loc,:) = Sampler(last_dist,options.d);
    loglike_curr(min_loc,:) = loglike_fn(theta_curr(min_loc,:),options);
    count_loglike = count_loglike + 1;
    
     %fprintf('Current val: %.2e , max_val: %.2e, ratio: %.2e, distance: %.2e \n', L_t, L_max, exp(L_t-L_max), last_dist)
end

%log_weights = log_weights - log_evidence;
% log_weights = log_weights - logsumexp(log_weights);
t = t+1;
log_evidence = logsumexp([log_evidence logsumexp(loglike_curr - t/N) - log(N)]);

prob = log(((N-1)/N)^t);
log_evidence_adj = logsumexp([log_evidence_adj, logsumexp(loglike_curr + prob - log(N))]);
end