function [theta, log_weights, log_evidence, count_loglike, levels, std_pop, R, w] = stratified_SLICE(loglike_fn,logprior_fn,simprior_fn,options,verbose)
% Stratified SMC with slice sampling move steps (inbuilt Matlab slice
% sampling)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function
%
% logprior_fn    -  log prior function
%
% simprior_fn    -  function to simulate from the prior
%
% options        -  options.N: Size of population of particles
%                -  options.rho: For choosing schedule
%                -  options.prob_move: Move probability for repeat calculation
%                -  options.choice_w: a set of w values, where w is the
%                   multiplicative factor for slice sampling
%                -  options.stopping_propZ: minimum proportion of the final
%                   log evidence to stop at. 1 is the (unreachable) gold standard.
%                -  options.stopping_ESS: minimum final ESS.
%                -  options.stopping_number: the allowable number of iterations.
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
% std_pop        -  The population standard deviations
%
% R              -  The number of MCMC repeats
%
% w              -  The multiplicative factor for slice sampling

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
rho = options.rho;
prob_move = options.prob_move;
choice_w = options.choice_w;
stopping_propZ = options.stopping_propZ;
stopping_ESS = options.stopping_ESS;
stopping_number = options.stopping_number;

theta_curr = simprior_fn(N,options);
d = size(theta_curr,2);

%initialise
log_evidence = -inf;
t = 1;
w = 1;

terminate = false;
theta = [];
log_weights = [];
levels = -inf;
prod_c = 1;
m = 0;

prop_Z = 0;
ESS_early = 0;

loglike_curr = zeros(N,1);
parfor i=1:N
    loglike_curr(i) = loglike_fn(theta_curr(i,:),options);
end

count_loglike = ones(N,1);

while ~terminate
    t = t+1;
    levels(t) = quantile(loglike_curr,(1-rho));
    
    %---- decide whether or not to terminate -----
    if t==stopping_number || prop_Z>stopping_propZ || ESS_early>=stopping_ESS
        terminate = true;
        levels(t) = Inf; % forces final strata
    end
    %------------------------------------------
    
    el_ind = (loglike_curr>=levels(t)); %indices of elite set
    
    num_elites = sum(el_ind);
    c = num_elites/N;
    
    if c == 1
        error('All Elites: Too many levels / Particles not moving')
    end
    
    %--------- Early finish stats ---------
    log_weights_early = [log_weights; loglike_curr + log(prod_c)];
    log_evidence_early = logsumexp(log_weights_early) - log(N);
    ESS_early = exp(2*logsumexp(log_weights_early)-logsumexp(2*log_weights_early));
    
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
    
    prop_Z = exp(log_evidence-log_evidence_early); % can tune to get this ~=1. Can also tune to get ESS_early/ESS_current~=1
    
    if verbose
        fprintf('\nIter %d\tLevel: %.4f\n\t\tEarly ESS: %.4f\n\t\tEarly log Z: %.4f\n\t\tProportion of Early Z: %.4f\n\t\tTotal elite: %d\n',t,levels(t),ESS_early,log_evidence_early,prop_Z,num_elites);
    end
    
    prod_c = prod_c*c;
    
    % Resample and Move
    % Note --- all the weights are equal so we just sample uniformly
    if levels(t) ~= Inf
        cov_part = cov(theta_curr(el_ind,:));
        inv_cov_part = inv(cov_part);
        [~,dists] = rangesearch(theta_curr(el_ind,:),theta_curr(el_ind,:),inf,'distance','mahalanobis','cov',cov_part);
        dists = cell2mat(dists);
        desired_dist = mean(dists(:));
    
        resamp_weights = zeros(N,1);
        resamp_weights(el_ind) = ones(num_elites,1)/(num_elites);
        inds = resampleStratified(resamp_weights);
        std_pop{t} = std(theta_curr,resamp_weights);
        
        if t==2 %need to evaluate log prior for the first time
            logprior_curr = zeros(N,1);
            for i=unique(inds)
                logprior_curr(i) = logprior_fn(theta_curr(i,:),options);
            end
        end
        
        theta_curr = theta_curr(inds,:);
        loglike_curr = loglike_curr(inds);
        logprior_curr = logprior_curr(inds);
        
        w_ind = mod(randperm(N),length(choice_w))'+1;
        w_all = choice_w(w_ind);
        ESJD = zeros(N,1);
        
        logtarget = @(x) logtarget_fn(x,levels(t),loglike_fn,logprior_fn,options);
        
        % Choosing the best w
        count_loglike_new = zeros(N,1);
        parfor i=1:N
            [theta_new, count_loglike_new(i)] = slicesample(theta_curr(i,:),1,'logpdf',logtarget,'width',w_all(i)*std_pop{t});
            ESJD(i) = (theta_curr(i,:)-theta_new)*inv_cov_part*(theta_curr(i,:)-theta_new)';
            theta_curr(i,:) = theta_new;
            loglike_curr(i,:) = loglike_fn(theta_curr(i,:),options);
            logprior_curr(i,:) = logprior_fn(theta_curr(i,:),options);
        end
        
        median_ESJD_per_count = accumarray(w_ind,ESJD./count_loglike_new,[],@median);
        count_loglike = count_loglike + count_loglike_new;
        
        [val, loc] = max(median_ESJD_per_count);
        w(t) = choice_w(loc);
        if val==0
            w(t) = w(t-1);
            fprintf('DID ALL ZERO FIX');
        end
        
        if verbose
            if mod(t,16)==0
                figure;
            end
            subplot(4,4,mod(t,16)+1);
            scatter(choice_w,median_ESJD_per_count); xlabel('w'); ylabel('Median ESJD/eval');
            hold on;
            scatter(w(t),median_ESJD_per_count(loc));
            drawnow;% pause(3);
            fprintf('\t\tThe new w is %d.\n',w(t));
        end
        
        % Performing the repeats with this w
        dist_move = zeros(N,1);
        belowThreshold = true;
        R(t) = 0;
        while belowThreshold
            R(t) = R(t) + 1;
            parfor i=1:N
                [theta_new, count_loglike_new] = slicesample(theta_curr(i,:),1,'logpdf',logtarget,'width',w(t)*std_pop{t});
                dist_move(i) = dist_move(i) + sqrt((theta_curr(i,:)-theta_new)*inv_cov_part*(theta_curr(i,:)-theta_new)');
                count_loglike(i) = count_loglike(i) + count_loglike_new;
                theta_curr(i,:) = theta_new;
                loglike_curr(i,:) = loglike_fn(theta_curr(i,:),options);
                logprior_curr(i,:) = logprior_fn(theta_curr(i,:),options);
            end
            if sum(dist_move>desired_dist)>=ceil(prob_move*N)
                belowThreshold = false;
            end
        end
        
        fprintf('\t\tThe number of MCMC repeats is %d.\n',R(t));
        
    end
    
end

%fprintf('Log evidence can also be calculated to be %.4f using the final weights.\n',logsumexp(log_weights)-log(N));
log_weights = log_weights - logsumexp(log_weights);

count_loglike = sum(count_loglike);

end
