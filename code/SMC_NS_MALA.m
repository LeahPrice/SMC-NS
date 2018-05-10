function [theta, log_weights, log_evidence, count_loglike, levels, R, cov_part, h] = stratified_MALA(loglike_fn,der_logprior_fn,simprior_fn,options,verbose)
% Stratified SMC with MH-MCMC MALA move steps.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike_fn     -  log likelihood function
%
% der_logprior_fn - derivative of the log prior function
%
% simprior_fn    -  function to simulate from the prior
%
% options        -  options.N: Size of population of particles
%                -  options.rho: For choosing schedule
%                -  options.prob_move: Move probability for repeat calculation
%                -  options.choice_h: a set of h values, where the
%                   multiplicative factor for the covariance is h^2
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
% R              -  The number of MCMC repeats required for the move step
%                   at each target
%
% cov_part       -  The covariance of the particles used in the move step
%
% h              -  The chosen values for h, where h^2 is the
%                   multiplicative factor for the covariance

if isa(loglike_fn,'function_handle') == 0
    loglike_fn = str2func(loglike_fn);
end
if isa(der_logprior_fn,'function_handle') == 0
    der_logprior_fn = str2func(der_logprior_fn);
end
if isa(simprior_fn,'function_handle') == 0
    simprior_fn = str2func(simprior_fn);
end

N = options.N;
rho = options.rho;
prob_move = options.prob_move;
choice_h = options.choice_h;
stopping_propZ = options.stopping_propZ;
stopping_ESS = options.stopping_ESS;
stopping_number = options.stopping_number;

theta_curr = simprior_fn(N,options);
d = size(theta_curr,2);

%initialise
log_evidence = -inf;
t = 1;
h = 2.38/sqrt(size(theta_curr,2));

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
        cov_part{t} = cov(theta_curr(el_ind,:));
        inv_cov_part = inv(cov_part{t});
        
        [~,dists] = rangesearch(theta_curr(el_ind,:),theta_curr(el_ind,:),inf,'distance','mahalanobis','cov',cov_part{t});
        dists = cell2mat(dists);
        desired_dist = mean(dists(:));
        
        resamp_weights = zeros(N,1);
        resamp_weights(el_ind) = ones(num_elites,1)/num_elites;
        inds = resampleStratified(resamp_weights);
        
        if t==2 %need to evaluate log prior for the first time
            logprior_curr = zeros(N,1); der_logprior_curr = zeros(N,d);
            for i=unique(inds)
                [logprior_curr(i), der_logprior_curr(i,:)] = der_logprior_fn(theta_curr(i,:),options);
            end
        end
        
        theta_curr = theta_curr(inds,:);
        loglike_curr = loglike_curr(inds);
        logprior_curr = logprior_curr(inds);
        der_logprior_curr = der_logprior_curr(inds,:);
        
        h_ind = mod(randperm(N),length(choice_h))'+1;
        h_all = choice_h(h_ind);
        ESJD = zeros(N,1);
        
        %  Choosing the best h
        accept = zeros(N,1); % for acceptance rate
        parfor i=1:N
            %MALA
            mymean = theta_curr(i,:)' + h_all(i)^2/2*cov_part{t}*(der_logprior_curr(i,:)');
            theta_prop = mvnrnd(mymean,h_all(i)^2*cov_part{t});
            
            [logprior_prop, der_logprior_prop] = der_logprior_fn(theta_prop,options);
            
            mymean_prop = theta_prop' + h_all(i)^2/2*cov_part{t}*(der_logprior_prop');
            transition_totheta = log_mvnpdf(theta_curr(i,:)',mymean_prop,h_all(i)^2*cov_part{t});
            transition_toprop = log_mvnpdf(theta_prop',mymean,h_all(i)^2*cov_part{t});
            
            loglike_prop = loglike_fn(theta_prop,options);
            count_loglike(i) = count_loglike(i) + 1;
            
            alpha = min(1, exp(logprior_prop - logprior_curr(i) + transition_totheta - transition_toprop)*(loglike_prop>=levels(t)));
            
            ESJD(i) = (theta_curr(i,:)-theta_prop)*inv_cov_part*(theta_curr(i,:)-theta_prop)' * alpha;
            
            if alpha > rand
                accept(i) = 1;
                theta_curr(i,:) = theta_prop;
                loglike_curr(i) = loglike_prop;
                logprior_curr(i) = logprior_prop;
                der_logprior_curr(i,:) = der_logprior_prop;
            end
        end
        
        median_ESJD = accumarray(h_ind,ESJD,[],@median);
        
        [val, loc] = max(median_ESJD);
        h(t) = choice_h(loc);
        if val==0
            h(t) = h(t-1);
            fprintf('DID ALL ZERO FIX');
        end
        
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
                mymean = theta_curr(i,:)' + h(t)^2/2*cov_part{t}*(der_logprior_curr(i,:)');
                theta_prop = mvnrnd(mymean,h(t)^2*cov_part{t});
                
                loglike_prop = loglike_fn(theta_prop,options);
                count_loglike(i) = count_loglike(i) + 1;
                
                [logprior_prop, der_logprior_prop] = der_logprior_fn(theta_prop,options);
                
                mymean_prop = theta_prop' + h(t)^2/2*cov_part{t}*(der_logprior_prop');
                transition_totheta = log_mvnpdf(theta_curr(i,:)',mymean_prop,h(t)^2*cov_part{t});
                transition_toprop = log_mvnpdf(theta_prop',mymean,h(t)^2*cov_part{t});
                
                alpha = min(1, exp(logprior_prop - logprior_curr(i) + transition_totheta - transition_toprop)*(loglike_prop>=levels(t)));
                dist_move(i) = dist_move(i) + sqrt((theta_curr(i,:)-theta_prop)*inv_cov_part*(theta_curr(i,:)-theta_prop)') * alpha;
                
                if alpha > rand
                    loglike_curr(i) = loglike_prop;
                    logprior_curr(i) = logprior_prop;
                    der_logprior_curr(i,:) = der_logprior_prop;
                end
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