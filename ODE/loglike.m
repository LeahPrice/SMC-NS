function [loglike] = loglike(x,options)
%Gets the log likelihood for parameter x which has been transformed to real
%line.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x              -  Parameter theta transformed to real line
%
% options        -  any additional options (maybe empty)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike        -  Log likelihood corresponding to inputted parameter


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
x = exp(x);
x(10) = true(10);

tf = 0:3:57; %20 time points - see paper for how we chose this.

y0 = x(6:9); %Initial values of S, D, R and Rpp are parameters 6-9
ypsir2 = @(t,y)[-x(1)*y(1); x(1)*y(1); -x(2)*y(3)*y(1)/(x(3)+y(3))+x(5)*y(4)/(x(4)+y(4)); x(2)*y(3)*y(1)/(x(3)+y(3))-x(5)*y(4)/(x(4)+y(4))]; % The ODE model
[~, y_res] = ode15s(ypsir2,tf,y0); %ODE solver, 15s is more stable than 45
diff = options.y_true-y_res(:,4);
loglike=sum(-log(x(10))-log(sqrt(2*pi))-diff.^2/(2*x(10)^2));

end