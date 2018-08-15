function [log_res] = log_mvnpdf(x,mu,cov)
%Does a stable calculation of log(mvnpdf(x,mu,Sigma));
diff=mu-x;
k=length(diff);
log_res=-k/2*log(2*pi)-1/2*log(det(cov))-1/2*diff'*inv(cov)*diff;
end

