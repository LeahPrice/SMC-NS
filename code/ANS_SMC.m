function [theta, log_weights, log_evidence, count_loglike, levels, cov_part, prod_c] = ANS_SMC(loglike_fn,logprior_fn,simprior_fn,options,verbose)
% ANS-SMC with MH-MCMC RW move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function taking the samples and a list as input
%
% logprior_fn    -  log prior function taking the samples and a list as input
%
% simprior_fn    -  function to simulate from the prior taking the sample size and list as input
%
% options        -  options.N: Size of population of particles
%                -  options.alpha: Threshold so (1-alpha)*N samples are above the next threshold
%                -  options.stopping_epsilon: maximum amount that the estimated proportion 
%                   of remaining evidence can be. 0 is the (unreachable) gold standard.
%                -  options.h: Multiplicative factor for the covariance is h^2
%                -  options.R: Number of MCMC repeats per particle per SMC iteration
%                -  ... example specific data and options
%
% verbose        -  set to true to get running update of progress and false otherwise

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
% levels         -  The levels for nested sampling
%
% cov_part       -  The covariance of the particles used in the move step
%
% prod_c         -  Proportion of remaining prior mass

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
stopping_epsilon = options.stopping_epsilon;

theta_curr = simprior_fn(N,options);
cov_part{1} = cov(theta_curr);
d = size(theta_curr,2);

%initialise
log_evidence = -inf;
t = 1;
h = options.h;
R = options.R;

terminate = false;
theta = [];
log_weights = [];
levels = -inf;
prod_c_curr = 1;
prod_c = 1;
m = 0;

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
    
    logRt = logsumexp(loglike_curr(el_ind)) + log(prod_c_curr) - log(N);
    
    %------ Storing the new samples ------
    new_inds = m+1 : m+sum(~el_ind);
    theta(new_inds,:) = theta_curr(~el_ind,:);
    log_weights(new_inds,1) = logw;
    m = max(new_inds);
    % ------------------------------------
    
    prod_c_curr = prod_c_curr*c;
    prod_c(t) = prod_c_curr;

    if verbose
        fprintf('\nIter %d\tLevel: %.4f\n\t\tCurrent log Z: %.4f\n\t\tTotal elite: %d\n',t,levels(t),log_evidence,num_elites);
    end
    
    % Resample and Move
    % Note --- all the weights are equal so we just sample uniformly
    if levels(t) ~= Inf
        cov_part{t} = cov(theta_curr(el_ind,:));
        
        resamp_weights = el_ind/num_elites;
        inds = resampleMultinomial(resamp_weights);
        
        if t==2 %need to evaluate log prior for the first time
            logprior_curr = zeros(N,1);
            for i=unique(inds)
                logprior_curr(i) = logprior_fn(theta_curr(i,:),options);
            end
        end
        
        theta_curr = theta_curr(inds,:);
        loglike_curr = loglike_curr(inds);
        logprior_curr = logprior_curr(inds);
        
        cholcov = chol(cov_part{t},'lower');
        
        % Performing the repeats with this h
        for k=1:R
            for i=1:N
                theta_prop = theta_curr(i,:)' + h*cholcov*randn([d,1]);
                logprior_prop = logprior_fn(theta_prop,options);
                
                if exp(logprior_prop - logprior_curr(i)) > rand
                    loglike_prop = loglike_fn(theta_prop,options);
                    count_loglike = count_loglike + 1;
                    
                    if loglike_prop>levels(t)
                        theta_curr(i,:) = theta_prop;
                        loglike_curr(i) = loglike_prop;
                        logprior_curr(i) = logprior_prop;
                    end
                end
            end
        end
        
    end
    
    %---- decide whether or not to terminate -----
    if logRt < log(stopping_epsilon) + logsumexp([logRt log_evidence])
        terminate = true;
        T = t+1;
        levels(T) = Inf; % forces final strata
        el_ind = zeros(N,1);
    end
    %--------------------------------------
    
end

prod_c_curr = prod_c_curr*c;
prod_c(t) = prod_c_curr;

logw = loglike_curr + log(prod_c_curr);
term = logsumexp(logw) - log(N);
log_evidence = logsumexp([log_evidence term]);

%------ Storing the new samples ------
new_inds = m+1 : m+N;
theta(new_inds,:) = theta_curr;
log_weights(new_inds,1) = logw;
% ------------------------------------

log_weights = log_weights - logsumexp(log_weights);

end