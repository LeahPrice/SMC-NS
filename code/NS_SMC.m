function [theta, log_weights, log_evidence, count_loglike, error_flag] = NS_SMC(loglike_fn,logprior_fn,simprior_fn,options,verbose)
% NS-SMC with MH-MCMC RW move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function taking the samples and a list as input
%
% logprior_fn    -  log prior function taking the samples and a list as input
%
% simprior_fn    -  function to simulate from the prior taking the sample size and list as input
%
% options        -  options.N: Size of population of particles
%                -  options.levels: The fixed levels
%                -  options.h: Multiplicative factor for the covariance is h^2
%                -  options.R: Number of MCMC repeats per particle per SMC iteration
%                -  options.cov_part: fixed covariance matrices for RW
%                -  ... example specific data and options
%
% verbose        -  set to true to get running update of progress and false otherwise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% theta          -  Samples from the posterior (weights below)
%
% log_weights    -  Log weights for the samples
%
% log_evidence 	 -  The log evidence estimate
%
% count_loglike  -  The total log likelihood computations
%
% error_flag     -  Zero if it finishes to level T, -t if stops because 
%                   no particles are above the next threshold at level t
%                   and t if it stops early because all particles are above
%                   the next threshold at level t

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
levels = options.levels;
h = options.h;
R = options.R;
cov_part = options.cov_part;

theta_curr = simprior_fn(N,options);
d = size(theta_curr,2);

%initialise
log_evidence = -inf;
T = length(levels);

theta = [];
log_weights = [];
prod_c = 1;
m = 0;
error_flag = 0;

loglike_curr = zeros(N,1);
for i=1:N
    loglike_curr(i) = loglike_fn(theta_curr(i,:),options);
end
count_loglike = N;

t = 2;
while t<=T
    el_ind = (loglike_curr>levels(t)); %indices of elite set
    
    num_elites = sum(el_ind);
    c = num_elites/N;
    
    if c == 0 && t~=T
        error_flag = -1*t;
        t = T;
    end
    
    if c ~= 1
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
    else
        error_flag = t;
    end
    
    prod_c = prod_c*c;
    
    if verbose
        fprintf('\nIter %d\tLevel: %.4f\n\t\tCurrent log Z: %.4f\n\t\tTotal elite: %d\n',t,levels(t),log_evidence,num_elites);
    end
    
    % Resample and Move
    % Note --- all the weights are equal so we just sample uniformly
    if levels(t) ~= Inf
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
        
        for i=1:N
            for k=1:R
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
    t = t + 1;
    
end

log_weights = log_weights - logsumexp(log_weights);

end