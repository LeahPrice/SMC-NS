function test = logtarget_fn(x,level,loglike_fn,logprior_fn,options)
% This is a helper function for NS-SMC with slice sampling. It returns the log of (prior*indicator of whether likelihood is greater than or equal to the level)
if loglike_fn(x,options)>=level
    test = logprior_fn(x,options);
else
    test = -inf;
end
end