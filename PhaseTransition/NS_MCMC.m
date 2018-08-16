function [theta, log_weights, log_evidence, count_loglike, log_evidence_adj] = NS_MCMC(options)
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

N = options.N;
R = options.R;
d = options.d;
L_max=loglike_fn(options.c,options);

theta = []; log_weights = [];

%initialise
log_evidence = -inf;
log_evidence_adj = -inf;
t = 0;
L_t  = -inf;

theta_curr = simprior_fn(N,options);
loglike_curr = zeros(N,1);

for i=1:N
    loglike_curr(i) = loglike_fn(theta_curr(i,:),options);
end
count_loglike = N;

while exp(L_t-L_max) < 0.75
    t = t+1;
    
    [L_t, min_loc] = min(loglike_curr);
    %log_weight = L_t + log(exp(-(t-1)/N)-exp(-t/N)); %Riemann sum
    log_weight = L_t + log((exp(-(t-1)/N)-exp(-(t+1)/N))/2); %trapezoidal
    
    %log_evidence = logsumexp([log_evidence log_weight]);
    log_evidence = logplusexp(log_evidence,log_weight);
    
    % adjusted version
    log_weight_adj = L_t + log(((N-1)/N)^(t-1)) - log(N);
    %log_evidence_adj = logsumexp([log_evidence_adj log_weight_adj]);
    log_evidence_adj = logplusexp(log_evidence_adj, log_weight_adj);
    
    %------ Storing the new samples and weights ------
    %theta(t,:) = theta_curr(min_loc,:);
    %log_weights(t) = log_weight;
    % ------------------------------------------------
    
    choice=min_loc;
    while choice==min_loc
        choice=ceil(rand*N);
    end
    
    %  fprintf('Current val: %.2e , max_val: %.2e, ratio: %.2e, distance: %.2e \n', L_t, L_max, exp(L_t-L_max), norm(theta_curr(min_loc,:)))
    theta_curr(min_loc,:) = theta_curr(choice,:);
    loglike_curr(min_loc,:) = loglike_curr(choice,:);
    
    for k=1:R
        if rand < 0.5
            sig = options.sig;
        else
            sig = options.sig/4;
        end
        
        theta_prop = theta_curr(min_loc,:);
        loc = ceil(rand*d);
        theta_prop(loc) = theta_prop(loc) + sig*randn;
        % theta_prop = theta_prop + sig * randn(1,d);
        if norm(theta_prop)<1
            loglike_prop = loglike_fn(theta_prop,options);
            count_loglike = count_loglike + 1;
            
            if loglike_prop > L_t
                theta_curr(min_loc,:) = theta_prop;
                loglike_curr(min_loc,:) = loglike_prop;
            end
            
        end
    end
end
%log_weights = log_weights - log_evidence;
%log_weights = log_weights - logsumexp(log_weights);

t = t+1;
log_evidence = logsumexp([log_evidence logsumexp(loglike_curr - t/N) - log(N)]);


prob = log(((N-1)/N)^t);

%exp(prob)

log_evidence_adj = logsumexp([log_evidence_adj, logsumexp(loglike_curr + prob - log(N))]);
end
