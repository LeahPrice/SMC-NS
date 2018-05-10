function [theta, loglike, logprior, log_evidence, count_loglike, gammavar, std_pop, R, w] = anneal_SLICE(loglike_fn,logprior_fn,simprior_fn,options,verbose)
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
%                -  options.alpha: For choosing schedule (ESS fixed at
%                   alpha*N.
%                -  options.prob_move: Move probability for repeat calculation
%                -  options.choice_w: a set of w values, where w is the
%                   multiplicative factor for slice sampling
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
%
% gammavar       -  The temperatures for the likelihood annealing schedule
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
alpha = options.alpha;
prob_move = options.prob_move;
choice_w = options.choice_w;
    
theta = simprior_fn(N,options);
d = size(theta,2);

% initialise
gammavar(1) = 0;
log_evidence = 0;
t = 1;

loglike = zeros(N,1);
logprior = zeros(N,1);
parfor i=1:N
    loglike(i) = loglike_fn(theta(i,:),options);
    logprior(i) = logprior_fn(theta(i,:),options); % some of these will be wasted - maybe move to later
end
count_loglike = ones(N,1);

while gammavar(t)~=1
    % Testing gammavar=1
    weights = (1-gammavar(t))*loglike(:,t);
    weights = weights-logsumexp(weights);
    ESS1 = exp(-logsumexp(2*weights));
    
    if (ESS1 >= alpha*N)
        gammavar(t+1) = 1;
    else
        gammavar(t+1) = bisection(@(thing)compute_ESS_diff(thing,gammavar(t),loglike(:,t),alpha*N),gammavar(t),1);
    end
    if verbose
        fprintf('The new temperature is %d.\n',gammavar(t+1));
    end
    
    logtarget = @(x) gammavar(t+1)*loglike_fn(x,options) + logprior_fn(x,options);
    
    weights = (gammavar(t+1)-gammavar(t))*loglike(:,t);
    log_evidence = log_evidence + logsumexp(weights) - log(N);
    weights = exp(weights-logsumexp(weights));
    theta_pop = theta(:,:,t);
    std_pop{t} = std(theta_pop,weights);
    
    cov_part = weightedcov(theta(:,:,t),weights); % covariance of resampled
    inv_cov_part = inv(cov_part);
    
    [~,dists] = rangesearch(theta(:,:,t),theta(:,:,t),inf,'distance','mahalanobis','cov',cov_part);
    dists = cell2mat(dists);
    tempmat = weights*weights';
    desired_dist = sum(tempmat(:).*dists(:)); % weighted mean
%     each_weighted = sum(bsxfun(@times,weights,dists'),1);
%     desired_dist = sum(weights'.*each_weighted);
    
    % Sampling with replacement according to weights
    ind = resampleStratified(weights);
    tplus1 = t + 1;
    theta(:,:,tplus1) = theta(ind,:,t);
    loglike(:,tplus1) = loglike(ind,t);
    logprior(:,tplus1) = logprior(ind,t);
    
    w_ind = mod(randperm(N),length(choice_w))'+1;
    w_all = choice_w(w_ind);
    ESJD = zeros(N,1);
    
    % Choosing the best w
    count_loglike_new = zeros(N,1);
    parfor i=1:N
        [theta_new, count_loglike_new(i)] = slicesample(theta(i,:,tplus1),1,'logpdf',logtarget,'width',w_all(i)*std_pop{t});
        ESJD(i) = (theta(i,:,tplus1)-theta_new)*inv_cov_part*(theta(i,:,tplus1)-theta_new)';
        theta(i,:,tplus1) = theta_new;
        loglike(i,tplus1) = loglike_fn(theta(i,:,tplus1),options);
        logprior(i,tplus1) = logprior_fn(theta(i,:,tplus1),options);
    end
    
    median_ESJD_per_count = accumarray(w_ind,ESJD./count_loglike_new,[],@median);
    count_loglike = count_loglike + count_loglike_new;
    
    [val, loc] = max(median_ESJD_per_count);
    w(t) = choice_w(loc);
    
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
            [theta_new, count_loglike_new] = slicesample(theta(i,:,tplus1),1,'logpdf',logtarget,'width',w(t)*std_pop{t});
            dist_move(i) = dist_move(i) + sqrt((theta(i,:,tplus1)-theta_new)*inv_cov_part*(theta(i,:,tplus1)-theta_new)');
            count_loglike(i) = count_loglike(i) + count_loglike_new;
            theta(i,:,tplus1) = theta_new;
            loglike(i,tplus1) = loglike_fn(theta(i,:,tplus1),options);
            logprior(i,tplus1) = logprior_fn(theta(i,:,tplus1),options);
        end
        if sum(dist_move>desired_dist)>=ceil(prob_move*N)
            belowThreshold = false;
        end
    end
    
    fprintf('\t\tThe number of MCMC repeats is %d.\n',R(t));
    
    t = tplus1;
end
count_loglike = sum(count_loglike);

end