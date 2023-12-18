function [log_evidence, count_loglike] = TA_SMC_bespoke_RW(loglike_fn,logprior_fn,simprior_fn,options,verbose)
% ATA-SMC with RW move steps.

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
%                -  options.sig: Standard deviation for mixture RW proposal (sig or sig/4)
%                -  ... example specific data and options
%
% verbose        -  set to true to get running update of progress and false otherwise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

N = options.N;
gammavar = options.gammavar;
R = options.R;

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
    
    for i=1:N
        for k=1:R
            if rand < 0.5
                sig = options.sig;
            else
                sig = options.sig/4;
            end
            
            theta_prop = theta(i,:);
            loc = ceil(rand*d);
            theta_prop(loc) = theta_prop(loc) + sig*randn;
            
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