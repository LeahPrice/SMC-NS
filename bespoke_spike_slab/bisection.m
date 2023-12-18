function p = bisection(f,a,b)

% provide the equation you want to solve with R.H.S = 0 form. 
% Write the L.H.S by using inline function
% Give initial guesses.
% Solves it by method of bisection.
% A very simple code. But may come handy

% Copyright (c) 2011, Brato CHAKRABARTI 
% All rights reserved.

% Redistribution and use in source and binary forms, with or without 
% modification, are permitted provided that the following conditions are 
% met:

% * Redistributions of source code must retain the above copyright 
 % notice, this list of conditions and the following disclaimer. 
 % * Redistributions in binary form must reproduce the above copyright 
 % notice, this list of conditions and the following disclaimer in 
 % the documentation and/or other materials provided with the distribution

% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE 
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
% POSSIBILITY OF SUCH DAMAGE. 


if f(a)*f(b)>0 
    disp('Wrong choice bro')
else
    p = (a + b)/2;
    err = abs(f(p));
    while err > 1e-7
   if f(a)*f(p)<0 
       b = p;
   else
       a = p;          
   end
    p = (a + b)/2; 
   err = abs(f(p));
    end
end
