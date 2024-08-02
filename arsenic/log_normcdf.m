function res = log_normcdf(x)

% normcdf(x) is equivalent to 0.5*erfc(-x/sqrt(2))
% Using erfc trick from https://github.com/JuliaMath/SpecialFunctions.jl/blob/master/src/erf.jl

res = x;
 
% For erfc of positive value
res(x<0) = log(0.5) + log(erfcx(-x(x<0)./sqrt(2))) - x(x<0).^2/2;

% For erfc of negative (or zero) value
res(x>=0) = log(0.5) + log(erfc(-x(x>=0)./sqrt(2)));

end