function f = logsumexp(x)
% Performs a stable calculation of log(sum(exp(x)))
my_max  = max(x);
x = x - my_max;
f = my_max + log(sum(exp(x)));
end