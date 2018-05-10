function [result] = compute_ESS_diff(thing,gammavar,loglike,desired)
%for bisection
weight=(thing-gammavar).*loglike;
weight=weight-max(weight);
weight=exp(weight);
weight=weight/sum(weight);
result=1/sum(weight.^2)-desired;

end

