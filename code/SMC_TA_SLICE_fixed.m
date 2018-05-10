function [theta, loglike, logprior, log_evidence, count_loglike] = anneal_SLICE_fixed(loglike_fn,logprior_fn,simprior_fn,options,verbose)
% Likelihood annealing SMC with slice sampling move steps (inbuilt Matlab slice
% sampling)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function
%
% logprior_fn    -  log prior function
%
% simprior_fn    -  function to simulate from the prior
%
% options        -  options.N: Size of population of particles
%                -  options.gammavar: The fixed temperatures
%                -  options.std_pop: The fixed population standard
%                   deviations for slice sampling
%                -  options.R: The fixed number of slice repeats
%                -  options.w: fixed multiplicative factors for slice
%                   sampling
%                -  ... example specific data and options
%
% verbose        -  set to true to get running update of progress and false otherwise

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
gammavar = options.gammavar;
std_pop = options.std_pop;
R = options.R;
w = options.w;

theta = simprior_fn(N,options);
d = size(theta,2);

% initialise
log_evidence = 0;
T = length(gammavar);

loglike = zeros(N,1);
logprior = zeros(N,1);
parfor i=1:N
    loglike(i) = loglike_fn(theta(i,:),options);
    logprior(i) = logprior_fn(theta(i,:),options); % some of these will be wasted - maybe move to later
end
count_loglike = ones(N,1);

for t=1:T-1
    logtarget = @(x) gammavar(t+1)*loglike_fn(x,options) + logprior_fn(x,options);
    
    weights = (gammavar(t+1)-gammavar(t))*loglike(:,t);
    log_evidence = log_evidence + logsumexp(weights) - log(N);
    weights = exp(weights-logsumexp(weights));
    
    % Sampling with replacement according to weights
    ind = resampleStratified(weights);
    tplus1 = t + 1;
    theta(:,:,tplus1) = theta(ind,:,t);
    loglike(:,tplus1) = loglike(ind,t);
    logprior(:,tplus1) = logprior(ind,t);
    
    % Performing an iteration with this w
    parfor i=1:N
        for z=1:R(t)
            [theta(i,:,tplus1), count_loglike_new] = slicesample(theta(i,:,tplus1),1,'logpdf',logtarget,'width',w(t)*std_pop{t});
            count_loglike(i) = count_loglike(i) + count_loglike_new;
            loglike(i,tplus1) = loglike_fn(theta(i,:,tplus1),options);
            logprior(i,tplus1) = logprior_fn(theta(i,:,tplus1),options);
        end
    end
end
count_loglike = sum(count_loglike);

end