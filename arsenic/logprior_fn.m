function logprior = logprior_fn(x,options)

[a, ~] = size(x);
if a==1
    x = x';
end

d = options.d;
logprior = log_mvnpdf(x,zeros(d,1),100*eye(d));


end