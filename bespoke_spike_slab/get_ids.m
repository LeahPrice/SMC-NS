function [threshL, el_ind, under_ind, over_ind, M] = get_ids(L,U,alpha)
% Returns the indices of the couples (L^k, U^k) lying above and below
% some order statistic

%threshL is the new  threshold level for L 
% el_ind is a binary vector of whether sample k is under or equal to (0), 
% or strictly above (1) the threshold
% under_ind:  the indices of samples underneath threshold or equal to it
% over_ind: the indices of samples over (i.e., which sample numbers have
% el_ind = 1) 

% Input is L: likelihood values, U: uniform random numbers, and
% alpha: adaptation parameter

%L and U are column vectors
N = size(L,1);
assert(size(L,1) == size(U,1))

% [sample id, L, U]
M = [(1:N)', L, U];

ostat_id = floor(N*(1-alpha));

% order first by L 
[L_orderedvals, orderL] = sort(L, 'ascend');
M = M(orderL,:);

% identify the L-threshold
threshL = L_orderedvals(ostat_id);

% --------- sort by U for the ties on the threshold ---------
% identify samples on the L-threshold

%rows with L on threshold
id_thresh = (M(:,2)==threshL);

if length(id_thresh) > 1
    % collection of U values for those rows
    thresh_subset_U = M(id_thresh,3);

    % sort those on the threshold by their U-values
    [~, orderU_thresh] = sort(thresh_subset_U, 'ascend');

    Mid_thresh = M(id_thresh,:); % subset to be reordered

     % reorder the subset according to U
    M(id_thresh,:) = Mid_thresh(orderU_thresh,:);
end
% ----------------------------------------------------------
final_order = M(:,1);

under_ind = final_order(1:ostat_id);
over_ind  = final_order((ostat_id+1):end);

el_ind = zeros(N,1);
el_ind(over_ind) = 1;
el_ind = logical(el_ind);

end
