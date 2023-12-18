function logprior = logprior_fn(x,options)

if norm(x)<1
    logprior = 0;
else
    logprior = -Inf;
end

end