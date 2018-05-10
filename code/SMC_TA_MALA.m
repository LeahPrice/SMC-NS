function [theta, loglike, logprior, der_loglike, der_logprior, log_evidence, count_loglike, gammavar, R, cov_part, h] = anneal_MALA(der_loglike_fn,der_logprior_fn,simprior_fn,options,verbose)
% Likelihood annealing SMC with MH-MCMC MALA move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% der_loglike_fn - derivative of the log likelihood function
%
% der_logprior_fn- derivative of the log prior function
%
% simprior_fn    -  function to simulate from the prior
%
% options        -  options.N: Size of population of particles
%                -  options.alpha: For choosing schedule (ESS fixed at
%                   alpha*N.
%                -  options.prob_move: Move probability for repeat calculation
%                -  options.choice_h: a set of h values, where the
%                   multiplicative factor for the covariance is h^2
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
% der_loglike    -  Derivative of the log likelihood corresponding to above thetas.
%
% der_logprior   -  Derivative of the log prior corresponding to above thetas.
%
% log_evidence 	 -  The log evidence estimate (not necessarily unbiased when
%                   adapting the temperatures and proposals adaptively
%                   online)
%
% count_loglike  -  The total log likelihood computations
%
% gammavar       -  The temperatures for the likelihood annealing schedule
%
% R              -  The number of MCMC repeats required for the move step
%                   at each temperature
%
% cov_part       -  The covariance of the particles used in the move step
%
% h              -  The chosen values for h, where h^2 is the
%                   multiplicative factor for the covariance

if isa(der_loglike_fn,'function_handle') == 0
    der_loglike_fn = str2func(der_loglike_fn);
end
if isa(der_logprior_fn,'function_handle') == 0
    der_logprior_fn = str2func(der_logprior_fn);
end
if isa(simprior_fn,'function_handle') == 0
    simprior_fn = str2func(simprior_fn);
end

N = options.N;
alpha = options.alpha;
prob_move = options.prob_move;
choice_h = options.choice_h;

theta = simprior_fn(N,options);
d = size(theta,2);

% initialise
gammavar(1) = 0;
log_evidence = 0;
t = 1;

loglike = zeros(N,1); der_loglike = zeros(N,d);
logprior = zeros(N,1); der_logprior = zeros(N,d);
parfor i=1:N
    [loglike(i), der_loglike(i,:)] = der_loglike_fn(theta(i,:),options); % some of the derivative calculations will be wasted...
    [logprior(i), der_logprior(i,:)] = der_logprior_fn(theta(i,:),options); % some of these will be wasted - maybe move to later
end

while gammavar(t)~=1
    % Testing gammavar=1
    w = (1-gammavar(t))*loglike(:,t);
    w = w-logsumexp(w);
    ESS1 = exp(-logsumexp(2*w));
    
    if (ESS1 >= alpha*N)
        gammavar(t+1) = 1;
    else
        gammavar(t+1) = bisection(@(thing)compute_ESS_diff(thing,gammavar(t),loglike(:,t),alpha*N),gammavar(t),1);
    end
    if verbose
        fprintf('The new temperature is %d.\n',gammavar(t+1));
    end
    
    w = (gammavar(t+1)-gammavar(t))*loglike(:,t);
    log_evidence = log_evidence + logsumexp(w) - log(N);
    w = exp(w-logsumexp(w));
    
    cov_part{t} = weightedcov(theta(:,:,t),w); % covariance of resampled
    inv_cov_part = inv(cov_part{t});
    
    [~,dists] = rangesearch(theta(:,:,t),theta(:,:,t),inf,'distance','mahalanobis','cov',cov_part{t});
    dists = cell2mat(dists);
    tempmat = w*w';
    desired_dist = sum(tempmat(:).*dists(:)); % weighted mean
%     each_weighted = sum(bsxfun(@times,w,dists'),1);
%     desired_dist = sum(w'.*each_weighted);
    
    % Sampling with replacement according to weights
    h_ind = resampleStratified(w);
    tplus1 = t + 1;
    theta(:,:,tplus1) = theta(h_ind,:,t);
    loglike(:,tplus1) = loglike(h_ind,t);
    der_loglike(:,:,tplus1) = der_loglike(h_ind,:,t);
    logprior(:,tplus1) = logprior(h_ind,t);
    der_logprior(:,:,tplus1) = der_logprior(h_ind,:,t);
        
    h_ind = mod(randperm(N),length(choice_h))'+1;
    h_all = choice_h(h_ind);
    ESJD = zeros(N,1);
    
    % Choosing the best h
    parfor i=1:N
        %MALA
        mymean = theta(i,:,tplus1)' + h_all(i)^2/2*cov_part{t}*(gammavar(t+1)*der_loglike(i,:,tplus1)'+der_logprior(i,:,tplus1)');
        theta_prop = mvnrnd(mymean,h_all(i)^2*cov_part{t});
        
        [loglike_prop, der_loglike_prop] = der_loglike_fn(theta_prop,options);
        [logprior_prop, der_logprior_prop] = der_logprior_fn(theta_prop,options);
        
        mymean_prop = theta_prop' + h_all(i)^2/2*cov_part{t}*(gammavar(t+1)*der_loglike_prop'+der_logprior_prop');
        transition_totheta = log_mvnpdf(theta(i,:,tplus1)',mymean_prop,h_all(i)^2*cov_part{t});
        transition_toprop = log_mvnpdf(theta_prop',mymean,h_all(i)^2*cov_part{t});
        
        log_mh = gammavar(t+1)*loglike_prop - gammavar(t+1)*loglike(i,tplus1) + logprior_prop - logprior(i,tplus1) + transition_totheta - transition_toprop;
        
        ESJD(i) = (theta(i,:,tplus1)-theta_prop)*inv_cov_part*(theta(i,:,tplus1)-theta_prop)' * min(1,exp(log_mh));
        
        if exp(log_mh) > rand
            theta(i,:,tplus1) = theta_prop;
            loglike(i,tplus1) = loglike_prop;
            der_loglike(i,:,tplus1) = der_loglike_prop;
            logprior(i,tplus1) = logprior_prop;
            der_logprior(i,:,tplus1) = der_logprior_prop;
        end
    end
    
    median_ESJD = accumarray(h_ind,ESJD,[],@median);
    
    [val, loc] = max(median_ESJD);
    h(t) = choice_h(loc);
    
    if verbose
        if mod(t,16)==0
            figure;
        end
        subplot(4,4,mod(t,16)+1);
        scatter(choice_h,median_ESJD); xlabel('h'); ylabel('Median ESJD');
        hold on;
        scatter(h(t),median_ESJD(loc));
        drawnow;% pause(3);
        fprintf('\t\tThe new h is %d.\n',h(t));
    end
    
    % Performing the repeats with this h
    dist_move = zeros(N,1);
    belowThreshold = true;
    R(t) = 0;
    while belowThreshold
        R(t) = R(t) + 1;
        parfor i=1:N
            %MALA
            mymean = theta(i,:,tplus1)' + h(t)^2/2*cov_part{t}*(gammavar(t+1)*der_loglike(i,:,tplus1)'+der_logprior(i,:,tplus1)');
            theta_prop = mvnrnd(mymean,h(t)^2*cov_part{t});
            
            [loglike_prop, der_loglike_prop] = der_loglike_fn(theta_prop,options);
            [logprior_prop, der_logprior_prop] = der_logprior_fn(theta_prop,options);
            
            mymean_prop = theta_prop' + h(t)^2/2*cov_part{t}*(gammavar(t+1)*der_loglike_prop'+der_logprior_prop');
            transition_totheta = log_mvnpdf(theta(i,:,tplus1)',mymean_prop,h(t)^2*cov_part{t});
            transition_toprop = log_mvnpdf(theta_prop',mymean,h(t)^2*cov_part{t});
        
            log_mh = gammavar(t+1)*loglike_prop - gammavar(t+1)*loglike(i,tplus1) + logprior_prop - logprior(i,tplus1) + transition_totheta - transition_toprop;
            
            dist_move(i) = dist_move(i) + sqrt((theta(i,:,tplus1)-theta_prop)*inv_cov_part*(theta(i,:,tplus1)-theta_prop)') * min(1,exp(log_mh));
            
            if exp(log_mh) > rand
                theta(i,:,tplus1) = theta_prop;
                loglike(i,tplus1) = loglike_prop;
                der_loglike(i,:,tplus1) = der_loglike_prop;
                logprior(i,tplus1) = logprior_prop;
                der_logprior(i,:,tplus1) = der_logprior_prop;
            end
        end
        if sum(dist_move>desired_dist)>=ceil(prob_move*N)
            belowThreshold = false;
        end
    end
    
    fprintf('\t\tThe number of MCMC repeats is %d.\n',R(t));
    
    t = tplus1;
end
count_loglike = N+N*sum(R+1);

end