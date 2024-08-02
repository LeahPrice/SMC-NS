function samples = simprior_fn(N,options)

d = options.d;
samples = mvnrnd(zeros(d,1),100*eye(d),N);

end