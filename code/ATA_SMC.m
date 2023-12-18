function [theta, log_evidence, count_loglike, gammavar, cov_part] = ATA_SMC(loglike_fn,logprior_fn,simprior_fn,options,verbose)
% ATA-SMC with MH-MCMC RW move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function taking the samples and a list as input
%
% logprior_fn    -  log prior function taking the samples and a list as input
%
% simprior_fn    -  function to simulate from the prior taking the sample size and list as input
%
% options        -  options.N: Size of population of particles
%                -  options.alpha: Threshold so ESS at each step is alpha*N
%                -  options.h: Multiplicative factor for the covariance is h^2
%                -  options.R: Number of MCMC repeats per particle per SMC iteration
%                -  ... example specific data and options
%
% verbose        -  set to true to get running update of progress and false otherwise
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theta          -  Samples from the posterior
%
% log_evidence 	 -  The log evidence estimate (not necessarily unbiased when
%                   adapting the temperatures and proposals adaptively
%                   online)
%
% count_loglike  -  The total log likelihood computations
%
% gammavar       -  The inverse temperatures for the likelihood annealing schedule
%
% cov_part       -  The covariance of the particles used in the move step

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
alpha = options.alpha;

theta = simprior_fn(N,options);
d = size(theta,2);

h = options.h;
R = options.R;

% initialise
gammavar(1) = 0;
log_evidence = 0;
t = 1;

loglike = zeros(N,1);
logprior = zeros(N,1);
for i=1:N
    loglike(i) = loglike_fn(theta(i,:),options);
    logprior(i) = logprior_fn(theta(i,:),options); % some of these will be wasted - maybe move to later
end
count_loglike = N;

while gammavar(t)~=1
    % Testing gammavar=1
    w = (1-gammavar(t))*loglike;
    w = w-logsumexp(w);
    ESS1 = exp(-logsumexp(2*w));
    
    if (ESS1 > alpha*N)
        gammavar(t+1) = 1;
    else
        gammavar(t+1) = bisection(@(thing)compute_ESS_diff(thing,gammavar(t),loglike,alpha*N),gammavar(t),1);
    end
    
    w = (gammavar(t+1)-gammavar(t))*loglike;
    log_evidence = log_evidence + logsumexp(w) - log(N);
    w = exp(w-logsumexp(w));
    
    if verbose
        fprintf('\nIter %d\tInverse temperature: %.4f\n\t\tCurrent log Z: %.4f\n',t+1,gammavar(t+1),log_evidence);
    end
    
    cov_part{t} = weightedcov(theta,w); % covariance of resampled
    
    % Sampling with replacement according to weights
    ind = resampleMultinomial(w);
    theta = theta(ind,:);
    loglike = loglike(ind);
    logprior = logprior(ind);
    
    cholcov = chol(cov_part{t},'lower');
    
    for k=1:R
        for i=1:N
            %MVN RW
            theta_prop = theta(i,:)' + h*cholcov*randn([d,1]);
            
            loglike_prop = loglike_fn(theta_prop,options);
            logprior_prop = logprior_fn(theta_prop,options);
            
            log_mh = gammavar(t+1)*(loglike_prop - loglike(i)) + logprior_prop - logprior(i);
            
            if exp(log_mh) > rand
                theta(i,:) = theta_prop;
                loglike(i) = loglike_prop;
                logprior(i) = logprior_prop;
            end
        end
    end
    
    t = t+1;
    count_loglike = count_loglike + N*R;
end

end