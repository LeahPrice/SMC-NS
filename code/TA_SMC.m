function [theta, log_evidence, count_loglike] = TA_SMC(loglike_fn,logprior_fn,simprior_fn,options,verbose)
% TA-SMC with MH-MCMC RW move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function taking the samples and a list as input
%
% logprior_fn    -  log prior function taking the samples and a list as input
%
% simprior_fn    -  function to simulate from the prior taking the sample size and list as input
%
% options        -  options.N: Size of population of particles
%                -  options.gammavar: The inverse temperatures for the likelihood annealing schedule
%                -  options.h: Multiplicative factor for the covariance is h^2
%                -  options.R: Number of MCMC repeats per particle per SMC iteration
%                -  options.cov_part: fixed covariance matrices for RW
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
gammavar = options.gammavar;
h = options.h;
R = options.R;
cov_part = options.cov_part;

theta = simprior_fn(N,options);
d = size(theta,2);

% initialise
log_evidence = 0;
T = length(gammavar);

loglike = zeros(N,1);
logprior = zeros(N,1);
for i=1:N
    loglike(i) = loglike_fn(theta(i,:),options);
    logprior(i) = logprior_fn(theta(i,:),options); % some of these will be wasted - maybe move to later
end
count_loglike = N;

for t=1:T-1
    w = (gammavar(t+1)-gammavar(t))*loglike;
    log_evidence = log_evidence + logsumexp(w) - log(N);
    w = exp(w-logsumexp(w));
    
    if verbose
        fprintf('\nIter %d\tInverse temperature: %.4f\n\t\tCurrent log Z: %.4f\n',t+1,gammavar(t+1),log_evidence);
    end
    
    % Sampling with replacement according to weights
    ind = resampleMultinomial(w);
    theta = theta(ind,:);
    loglike = loglike(ind);
    logprior = logprior(ind);
    
    cholcov = chol(cov_part{t},'lower');
    
    for i=1:N
        for k=1:R
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
    count_loglike = count_loglike + N*R;
end

end