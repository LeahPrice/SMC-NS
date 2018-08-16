function X=Sampler(dist,d)
%generates a point closer than distance 'dist' from the origin
%i.e., generate point in d-sphere of radius 'dist'
Y=randn(1,d);
U=rand;
R=dist*U^(1/d);
X=R.*(Y./norm(Y,2));
end
