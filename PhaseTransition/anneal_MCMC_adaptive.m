function [theta, loglike, log_evidence, count_loglike, gammavar] = anneal_MCMC_adaptive(options)
% Likelihood annealing SMC

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% options        -  options.N: Size of population of particles
%                -  options.alpha: For choosing temperatures
%                -  options.R: fixed number of MCMC repeats
%                -  options.sig: fixed parameter for sampler
%                -  options.d: dimension

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
%
% gammavar       -  Temperature schedule for annealing

N = options.N;
alpha = options.alpha;
R = options.R;
d = options.d;
sig = options.sig;

theta = simprior_fn(N,options);

% initialise
log_evidence = 0;
gammavar(1) = 0;
t = 1;

loglike = zeros(N,1);
for i=1:N
    loglike(i) = loglike_fn(theta(i,:),options);
end
count_loglike = N;

while gammavar(t)~=1
    % Testing gammavar=1
    w = (1-gammavar(t))*loglike(:,t);
    w = w-logsumexp(w);
    ESS1 = exp(-logsumexp(2*w));
    
    if (ESS1 >= alpha*N)
        gammavar(t+1) = 1;
    else
        gammavar(t+1) = bisection(@(thing)compute_ESS_diff(thing,gammavar(t),loglike(:,t),alpha*N),gammavar(t),1);
    end
    
    w = (gammavar(t+1)-gammavar(t))*loglike(:,t);
    log_evidence = log_evidence + logsumexp(w) - log(N);
    w = exp(w-logsumexp(w));
    
    % Sampling with replacement according to weights
    %ind = resampleStratified(w);
    ind = resampleMultinomial(w);
    tplus1 = t + 1;
    theta(:,:,tplus1) = theta(ind,:,t);
    loglike(:,tplus1) = loglike(ind,t);
    
    for i=1:N
        for k=1:R
            
            % theta_prop = mvnrnd(theta(i,:,tplus1), sig^2 * eye(d));
            if rand < 0.5
                sig = options.sig;
            else
                sig = options.sig/4;
            end
            
            
            theta_prop = theta(i,:,tplus1);
            
            ind = ceil(rand * d);
            theta_prop(ind) = theta_prop(ind) + sig * randn;
            loglike_prop = loglike_fn(theta_prop,options);
            count_loglike = count_loglike + 1;
            if norm(theta_prop,2) <=1
                log_mh = gammavar(t+1)*(loglike_prop - loglike(i,tplus1));
                
                if exp(log_mh) > rand
                    
                    theta(i,:,tplus1) = theta_prop;
                    loglike(i,tplus1) = loglike_prop;
                end
            end
        end
        
    end
    
    t = tplus1;
end
count_loglike = sum(count_loglike);
end
