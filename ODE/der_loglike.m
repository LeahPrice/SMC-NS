function [loglike, der_loglike] = der_loglike(x,options)
%Gets the derivative of the log likelihood wrt parameters x which have
%been transformed to real line.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x              -  Parameter theta transformed to real line
%
% options        -  any additional options (maybe empty)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike        -  Log likelihood corresponding to inputted parameter
%
% der_loglike    -  Derivative of log likelihood wrt parameters


% List of parameters in order
% 1  k1
% 2  V1
% 3  Km1
% 4  Km2
% 5  V2
% 6  S0
% 7  D0
% 8  R0
% 9 Rpp0
% 10 sigma

% True values that we have simulated from
true = [0.05; 0.2; 0.1; 0.1; 0.1; 1; 0; 1; 0; 0.02];

% Transforming back to original scale
theta = exp(x);
theta(10) = true(10);

tf = 0:3:57; %20 time points - see paper for how we chose this.

y0 = zeros(1,40);
y0(1:4) = theta(6:9); %Initial values of S, D, R and Rpp are parameters 6-9
y0([25 30 35 40]) = ones(1,4); % Derivative of responses wrt their initial values are in these spots
ypsir2 = @(t,y) myode(y,theta(1:5));
[~, y_res] = ode15s(ypsir2,tf,y0); %ODE solver, 15s is more stable than 45
temp = y_res(:,[20:24,37:40]);
der_loglike = theta(10)^-2 * sum(temp .*((options.y_true - y_res(:,4))* ones(1,9)),1);
der_loglike = der_loglike.*theta(1:9); % because we want derivative wrt transformed variable...
diff = options.y_true-y_res(:,4);
loglike = sum(-log(theta(10))-log(sqrt(2*pi))-diff.^2/(2*theta(10)^2));

end