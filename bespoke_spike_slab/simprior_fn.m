function samples = simprior_fn(N,options)

d = options.d;

samples = zeros(N,d);

for i=1:N
    samples(i,:) = bespoke_exact(1,d);
end

end
