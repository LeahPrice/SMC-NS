function [logprior, der_logprior] = der_logprior(x,options)

logprior = zeros(1,9);
logprior(1) = sum(x(1)-exp(x(1)),2);
logprior(2) = sum(x(2)-exp(x(2)),2);
logprior(3) = sum(x(3)-exp(x(3)),2);
logprior(4) = sum(x(4)-exp(x(4)),2);
logprior(5) = sum(x(5)-exp(x(5)),2);
logprior(6) = sum(x(6)-5*log(0.2)-log(gamma(5))+(5-1)*x(6)-exp(x(6))/0.2,2);
logprior(7) = sum(x(7)-1*log(0.1)-exp(x(7))/0.1,2);
logprior(8) = sum(x(8)-5*log(0.2)-log(gamma(5))+(5-1)*x(8)-exp(x(8))/0.2,2);
logprior(9) = sum(x(9)-1*log(0.1)-exp(x(9))/0.1,2);
logprior = sum(logprior);

der_logprior = zeros(1,9);
der_logprior(1) = sum(1-exp(x(1)),2);
der_logprior(2) = sum(1-exp(x(2)),2);
der_logprior(3) = sum(1-exp(x(3)),2);
der_logprior(4) = sum(1-exp(x(4)),2);
der_logprior(5) = sum(1-exp(x(5)),2);
der_logprior(6) = sum(5-exp(x(6))/0.2,2);
der_logprior(7) = sum(1-exp(x(7))/0.1,2);
der_logprior(8) = sum(5-exp(x(8))/0.2,2);
der_logprior(9) = sum(1-exp(x(9))/0.1,2);

end

