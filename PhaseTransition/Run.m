N = 1e3; % number of particles
R = 10; % number of MCMC repeats
runs = 100;
exact_sample = 0; % use exact sampler (0 for exact, 1 for MCMC)

p = fileparts(which('Run.m')); % getting the location of the 'Run.m' file
location = strcat(p(1:(strfind(p, '\PhaseTransition'))), 'code'); % getting the location of the 'code' folder - Windows
% location = strcat(p(1:(strfind(p, '/PhaseTransition'))), 'code'); % getting the location of the 'code' folder
addpath(genpath(location)); % adding the contents of the 'code' folder to the working directory

Run_fn(N,R,runs,exact_sample)
