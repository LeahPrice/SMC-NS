function [loglike] = loglike_fn(x,options)
% Gets the log likelihood (Gaussian mixture pdf) for parameter x

d = options.d;
u=0.1; v=0.01;

a1 = 0.1;
a2 = 0.9;

s = sum(x.^2);
c1 = log(a1) - d/2*log(2*pi*u^2) - 0.5*s/u^2;
c2 = log(a2) - d/2*log(2*pi*v^2) - 0.5*s/v^2;

mxm = max(c1,c2);
mnm = min(c1,c2);

loglike = mxm + log(1+ exp(mnm -mxm));

end