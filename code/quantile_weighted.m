function [quantile] = quantile_weighted(samples,p,weights)
% Gets quantiles for a weighted sample

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% samples        -  The sample particles (that have associated weights)
%
% p              -  Desired quantile
%
% weights        -  Weights associated with samples

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% quantile       -  The sample associated with desired quantile p.

[N, d] = size(samples);
quantile = zeros(1,d);

if length(weights)~=N
    error('The samples and weights should be of the same size!');
end

for j=1:d
    [sorted, order] = sort(samples(:,j));
    cumsum_w = weights(order);
    cumsum_w = cumsum(cumsum_w);
    lower = find(cumsum_w<p+10^(-15),1,'last');
    upper = lower + 1;
    quantile(j) = sorted(lower) + (p - cumsum_w(lower))/(cumsum_w(upper) - cumsum(lower))*(sorted(upper) - sorted(lower)); %Linear interpolation
end

end

