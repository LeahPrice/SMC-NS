function val = logplusexp(a,b)
mx = max(a,b); mn = min(a,b); 
val = mx + log(1+ exp(mn-mx)); 
end
