function Y=myode(X,U)

% See section 1.3.1 of http://faculty.bscb.cornell.edu/~hooker/profile_webpages/ODE_Estimation.pdf
% for general methods to getting the derivatives wrt parameters and 

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

% The original ODE system
Y(1) = -U(1)*X(1); % dS/dt
Y(2) = -Y(1); % dD/dt
Y(3) = -U(2)*X(3)*X(1)/(U(3)+X(3)) + U(5)*X(4)/(U(4)+X(4)); %dR/dt
Y(4) = -Y(3); %dRpp/dt

% Parameter sensitivities
Y(5) = - X(1)-U(1)*X(5); %dS/dk1 dt
Y(6) = -U(1)*X(6); %dS/dV1 dt
Y(7) = -U(1)*X(7); %dS/dKm1 dt
Y(8) = -U(1)*X(8); %dS/dKm2 dt
Y(9) = -U(1)*X(9); %dS/dV2 dt

Y(10) = -Y(5); %dD/dk1 dt
Y(11) = -Y(6); %dD/dV1 dt
Y(12) = -Y(7); %dD/dKm1 dt
Y(13) = -Y(8); %dD/dKm2 dt
Y(14) = -Y(9); %dD/dV2 dt

Y(15) = -U(2)*X(3)*X(5)/(U(3) + X(3)) - U(2)*X(15)*X(1)*U(3)/(U(3)+X(3))^2   +   U(5)*U(4)*X(20)/(U(4) + X(4))^2; % dR/dk1 dt
Y(16) = -X(3)*X(1)/(U(3)+X(3)) - U(2)*X(3)*X(6)/(U(3) + X(3)) - U(2)*X(16)*X(1)*U(3)/(U(3)+X(3))^2   +   U(5)*U(4)*X(21)/(U(4) + X(4))^2; % dR/dV1 dt
Y(17) = -U(2)*X(3)*X(7)/(U(3) + X(3)) - U(2)*X(1)*(X(17)*U(3)-X(3))/(U(3)+X(3))^2   +   U(5)*U(4)*X(22)/(U(4) + X(4))^2; % dR/dKm1 dt
Y(18) = -U(2)*X(3)*X(8)/(U(3) + X(3)) - U(2)*X(18)*X(1)*U(3)/(U(3)+X(3))^2   +   U(5)*(U(4)*X(23)-X(4))/(U(4) + X(4))^2; % dR/dKm2 dt
Y(19) = -U(2)*X(3)*X(9)/(U(3) + X(3)) - U(2)*X(19)*X(1)*U(3)/(U(3)+X(3))^2   +   X(4)/(U(4)+X(4)) + U(5)*U(4)*X(24)/(U(4) + X(4))^2; % dR/dV2 dt

Y(20) = -Y(15); %dRpp/dk1 dt
Y(21) = -Y(16); %dRpp/dV1 dt
Y(22) = -Y(17); %dRpp/dKm1 dt
Y(23) = -Y(18); %dRpp/dKm2 dt
Y(24) = -Y(19); %dRpp/dV2 dt

% Initial value sensitivities
Y(25) = -U(1)*X(25); %dS/dS0 dt
Y(26) = -U(1)*X(26); %dS/dD0 dt
Y(27) = -U(1)*X(27); %dS/dR0 dt
Y(28) = -U(1)*X(28); %dS/dRpp0 dt

Y(29) = -Y(25); %dD/dS0 dt
Y(30) = -Y(26); %dD/dD0 dt
Y(31) = -Y(27); %dD/dR0 dt
Y(32) = -Y(28); %dD/dRpp0 dt

Y(33) = -U(2)*X(3)*X(25)/(U(3) + X(3)) - U(2)*X(33)*X(1)*U(3)/(U(3)+X(3))^2   +   U(5)*U(4)*X(37)/(U(4) + X(4))^2; % dR/dS0 dt
Y(34) = -U(2)*X(3)*X(26)/(U(3) + X(3)) - U(2)*X(34)*X(1)*U(3)/(U(3)+X(3))^2   +   U(5)*U(4)*X(38)/(U(4) + X(4))^2; % dR/dD0 dt
Y(35) = -U(2)*X(3)*X(27)/(U(3) + X(3)) - U(2)*X(35)*X(1)*U(3)/(U(3)+X(3))^2   +   U(5)*U(4)*X(39)/(U(4) + X(4))^2; % dR/dR0 dt
Y(36) = -U(2)*X(3)*X(28)/(U(3) + X(3)) - U(2)*X(36)*X(1)*U(3)/(U(3)+X(3))^2   +   U(5)*U(4)*X(40)/(U(4) + X(4))^2; % dR/dRpp0 dt

Y(37) = -Y(33); %dRpp/dS0 dt
Y(38) = -Y(34); %dRpp/dD0 dt
Y(39) = -Y(35); %dRpp/dR0 dt
Y(40) = -Y(36); %dRpp/dRpp0 dt


Y = Y';

end