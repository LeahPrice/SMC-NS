function [loglike] = loglike_fn(x,options)
%Gets the log likelihood for parameter x which has been transformed to real
%line.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% x              -  Parameter theta transformed to real line
%
% options        -  any additional options (maybe empty)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% OUTPUT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% loglike        -  Log likelihood corresponding to inputted parameter


d = options.d;
u=0.1; v=0.01;
c = 0;


log_scale = 1; % calculate on log scale for stability, slower and doesn't seem to make a difference for this problem
a1 = 1/4;
a2 = 3/4;

if log_scale
    t1=log(a1); t2=log(a2);
    
    for i=1:d
        t1=t1+(-((x(i))^2)/(2*u^2));
        t2=t2+(-(x(i)-c)^2/(2*v^2));
    end
    
    c1 = d *(log(1) - log(sqrt(2*pi)*u));
    c2 = d *(log(1) - log(sqrt(2*pi)*v));
    
    tt1 = t1+c1;
    tt2 = t2+c2;
    
    mxm = max(tt1,tt2);
    mnm = min(tt1,tt2);
    
    loglike = mxm + log(1+ exp(mnm -mxm)); 
else
    t1=a1; t2=a2;
    for i=1:d
        t1=t1*exp(-((x(i))^2)/(2*u^2));
        t2=t2*exp(-(x(i)-c)^2/(2*v^2));
    end
    
    c1 = (1/(sqrt(2*pi)*u)).^d;
    c2 = (1/(sqrt(2*pi)*v)).^d;
    
    like = (t1*c1) + (t2*c2);
    loglike = log(like);
end



%loglike = logsumexp([t1+c1, t2+c2]);

% d = options.d;
%
% u=0.1; v=0.01;
% t1=1/101; t2=100/101;
% c = 0.025;
%
% for i=1:d
%     t1=t1*exp(-((x(i))^2)/(2*u^2));
%     t2=t2*exp(-(x(i)-c)^2/(2*v^2));
% end
%
% c1 = (1/(sqrt(2*pi)*u)).^d;
% c2 = (1/(sqrt(2*pi)*v)).^d;
%
% like = (t1*c1) + (t2*c2);
% loglike = log(like);
% end