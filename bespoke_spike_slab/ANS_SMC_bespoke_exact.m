function [log_evidence, count_loglike, levels, dists] = ANS_SMC_bespoke_exact(loglike_fn,simprior_fn,options,verbose)
% ANS-SMC with bespoke, exact sampling for the move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function taking the samples and a list as input
%
% logprior_fn    -  log prior function taking the samples and a list as input
%
% simprior_fn    -  function to simulate from the prior taking the sample size and list as input
%
% options        -  options.N: Size of population of particles
%                -  options.alpha: Threshold so (1-alpha)*N samples are above the next threshold
%                -  options.R: Number of MCMC repeats per particle per SMC iteration
%                -  ... example specific data and options
%
% verbose        -  set to true to get running update of progress and false otherwise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% log_evidence 	 -  The log evidence estimate (not necessarily unbiased when
%                   adapting the temperatures and proposals adaptively
%                   online)
%
% count_loglike  -  The total log likelihood computations
%
% levels         -  The levels for nested sampling
%
% dists          -  Maximum norm of the sphere for each level

if isa(loglike_fn,'function_handle') == 0
    loglike_fn = str2func(loglike_fn);
end
if isa(simprior_fn,'function_handle') == 0
    simprior_fn = str2func(simprior_fn);
end

N = options.N;
alpha = options.alpha;

theta_curr = simprior_fn(N,options);
dists(1) = 1;
d = size(theta_curr,2);
loglike0 = loglike_fn(zeros(1,d),options);

%initialise
log_evidence = -inf;
t = 1;

terminate = false;
levels = -inf;
prod_c_curr = 1;

U = rand(N,1); % random numbers for breaking ties

loglike_curr = zeros(N,1);
for i=1:N
    loglike_curr(i) = loglike_fn(theta_curr(i,:),options);
end
count_loglike = N;

while ~terminate
    t = t+1;
    
    [levels(t), el_ind, ~, ~,~] = get_ids(loglike_curr,U,alpha);
    
    num_elites = sum(el_ind);
    c = num_elites/N;
    
    if c == 1
        error('All Elites: Too many levels / Particles not moving')
    end
    
    %--- weighted (unnormalized) samples---
    logw = loglike_curr(~el_ind) + log(prod_c_curr);
    term = logsumexp(logw) - log(N); % Z_t
    log_evidence = logsumexp([log_evidence term]);
    
    prod_c_curr = prod_c_curr*c;
    
    if verbose
        fprintf('\nIter %d\tLevel: %.4f\n\t\tCurrent log Z: %.4f\n\t\tTotal elite: %d\n',t,levels(t),log_evidence,num_elites);
    end
    
    % Resample and Move
    % Note --- all the weights are equal so we just sample uniformly
    if levels(t) ~= Inf
        dists(t) = norm(theta_curr(loglike_curr == levels(t),:));
        
        resamp_weights = el_ind/num_elites;
        inds = resampleMultinomial(resamp_weights);
        
        theta_curr = theta_curr(inds,:);
        loglike_curr = loglike_curr(inds);
        
        for i=1:N
            theta_curr(i,:) = bespoke_exact(dists(t),options.d);
            loglike_curr(i,:) = loglike_fn(theta_curr(i,:),options);
            count_loglike = count_loglike + 1;
        end
        
    end
    
    %---- decide whether or not to terminate -----
    if levels(t) > (log(0.75) + loglike0)
        terminate = true;
        T = t+1;
        levels(T) = Inf; % forces final strata
        el_ind = zeros(N,1);
    end
    %--------------------------------------
    
end

prod_c_curr = prod_c_curr*c;

logw = loglike_curr + log(prod_c_curr);
term = logsumexp(logw) - log(N);
log_evidence = logsumexp([log_evidence term]);

end