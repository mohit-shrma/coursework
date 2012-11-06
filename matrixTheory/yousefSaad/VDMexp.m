 disp(' set data ') 
 n = 6;
 tau = 1.e-10;
 A = vander( [1:n] .^ 2);
 b = A*ones(n,1);
 F = randn(n,n); % f = randn(n,1);
 % load Rand6;
 E = A .*F ;
 e = b .* f; 
 B = A+tau*E; 
 c = b+tau*e; 
 y = B\c;
 x = ones(n,1); 
 disp(' actual rel. error')
 norm(y-x)/norm(x)
 pause 

 disp(' Theorem 2') 
 num=tau*cond(A)*(norm(e)/norm(b)+norm(E)/norm(A));
 denom = 1 - tau*norm(inv(A))*norm(E);
 num/denom
 pause 

 disp(' comp. cond. num.')
 norm(abs(inv(A))*abs(A),inf)
 %% cond(A,inf)
 pause 



 disp(' Theorem 3') 
 r = b-A*y;
 eta = norm(r)/(norm(E)*norm(y) + norm(e))
 pause


 disp(' Theorem 5') 
 %% 
 E = abs(E);
 e = abs(e);
 num = tau*norm(abs(inv(A))*E*abs(x)+e)/norm(x);
% num = tau*norm(abs(inv(A))*E*abs(y)+e)/norm(y);
 denom =  1 - tau * norm(abs(inv(A))*E);
 err = num/denom


 pause 
%% 
 disp(' Oettli-Prager')
 r = b - A*y;  % done above
 z = E*abs(y) + e;
 min(z);
 t = abs(r) ./ z;
 omega = max (t)

