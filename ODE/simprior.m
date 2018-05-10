function samples = simprior(N,options)

%Draws from prior
samples = [gamrnd(1,1,[N,1]) gamrnd(1,1,[N,1]) gamrnd(1,1,[N,1]) gamrnd(1,1,[N,1]) gamrnd(1,1,[N,1]) gamrnd(5,0.2,[N,1]) gamrnd(1,0.1,[N,1]) gamrnd(5,0.2,[N,1]) gamrnd(1,0.1,[N,1])];
samples = log(samples);

end

