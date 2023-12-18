function [log_evidence, count_loglike, error_flag] = NS_SMC_bespoke_exact(loglike_fn,simprior_fn,options,verbose)
% NS-SMC with bespoke, exact sampling for the move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function taking the samples and a list as input
%
% simprior_fn    -  function to simulate from the prior taking the sample size and list as input
%
% options        -  options.N: Size of population of particles
%                -  options.levels: The fixed levels
%                -  options.dists: maximum norm of the sphere corresponding to levels
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
% error_flag     -  Zero if it finishes to level T, -t if stops because 
%                   no particles are above the next threshold at level t
%                   and t if it stops early because all particles are above
%                   the next threshold at level t

if isa(loglike_fn,'function_handle') == 0
    loglike_fn = str2func(loglike_fn);
end
if isa(simprior_fn,'function_handle') == 0
    simprior_fn = str2func(simprior_fn);
end

N = options.N;
levels = options.levels;
dists = options.dists;

theta_curr = simprior_fn(N,options);

%initialise
log_evidence = -inf;
T = length(levels);

prod_c = 1;
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
        
        theta_curr = theta_curr(inds,:);
        loglike_curr = loglike_curr(inds);
        
        for i=1:N
            theta_curr(i,:) = bespoke_exact(dists(t),options.d);
            loglike_curr(i) = loglike_fn(theta_curr(i,:),options);
            count_loglike = count_loglike + 1;
        end
    end
    
    t = t + 1;
    
end

end