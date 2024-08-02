function [loglike] = loglike_fn(x,options)
% x coefficients
% X design matrix

[a, ~] = size(x);
if a==1
    x = x';
end

Xbeta = options.X*x;
loglike = sum(options.y.*log_normcdf(Xbeta) + (1-options.y).*log_normcdf(-Xbeta));

end