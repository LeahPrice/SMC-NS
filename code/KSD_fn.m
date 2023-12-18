function KSD = KSD_fn(samples, gr_samples, bandwidth)
%% K_PI Computes the Gram matrix of the Stein kernel:
%   
%      (  k_p(x_1,x_1)  ...  k_p(x_1,x_n)  )
%      (        .       ...        .       )
%      (  k_p(x_n,x_1)  ...  k_p(x_n,x_n)  )
%
%      under the inverse multiquadric (IMQ) kernel w/ dimension scaling.
%
%      For dimension d and no. of samples n:
%
%      -- samples is a n x d matrix containing n samples x_1,...,x_n
%         i.e. samples(i,:) = x_i is a single sample (each column)
%
%      -- gr_samples is a n x d matrix containing the gradient information
%         (score function) computed at each sample
%         i.e. gr_samples(i,:) = grad log p(x_i)
%
n = size(samples, 1);
d = size(samples, 2);

if nargin < 3
    bandwidth = 1;
end

norms = mat_dot_prod(samples, samples);
norms = norms - diag(diag(norms));

[k_norms,k_diff,k_diff2] = k(norms,-0.5,bandwidth);

k_gram = -2 * d * k_diff - 4 * norms.*k_diff2;
k_gram = k_gram - 2 * k_diff.*mat_dot_prod(samples, gr_samples);
k_gram = k_gram + (gr_samples * gr_samples').*k_norms;

%KSD = sum(sum(k_gram));
KSD = sqrt(sum(sum(k_gram)))/n;
end

function M = mat_dot_prod( mat1, mat2 )
%FAST_ABK_PROD Uses vectorised code to compute
%  ---
%  \   [  A   - A   ] [  B    -  B   ]
%  /   [   ik    jk ] [   ik      jk ]
%  ---
%   k

d = size(mat1,1);
assert(size(mat1,1) == size(mat2,1));
assert(size(mat1,2) == size(mat2,2));

norms = sum(mat1.*mat2, 2);
norms = repmat(norms, 1, d);
mul = mat1 * mat2';

M = norms + norms' - mul - mul';
M = M - diag(diag(M)); % Just in case

end

function [k0,k1,k2] = k(x,beta,bandwidth)
% Computes the IMQ kernel, scaled according to bandwidth
x = x / sqrt(bandwidth);
k0 = (1 + x).^(beta);
k1 = beta*(1+x).^(beta-1) / sqrt(bandwidth);
k2 = beta*(beta-1)*(1+x).^(beta-2) / bandwidth;

end

