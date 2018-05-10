function [log_res] = log_normpdf(x,mu,sigma)
%Does a stable calculation of log(normpdf(x,mu,sigma));
diff=x-mu;
log_res=-log(sigma)-log(sqrt(2*pi))-diff.^2/(2*sigma^2);

end

