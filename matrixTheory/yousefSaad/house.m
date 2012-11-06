 function [v,bet] = house(x) 
%% function [v,bet] = house (x) 
%% computes the householder vector to 
%% introduce zeros in locations 2 to m of x
x = x(:);
m = length(x);
%%-------------------- trivial case
if (m == 1) v=1;, bet=0;, return;, end;
v = [1 ; x(2:m)]; 
sigma = v(2:m)' * v(2:m);
if (sigma == 0) 
    bet = 0;
else
    alpha = sqrt(x(1)^2 + sigma) ;
    if (x(1) <= 0)
        v(1) = x(1) - alpha;
    else 
        v(1) = x(1) + alpha; 
    end
%%-------------------- normalize so v(1)=1
    bet = 2 / (1+sigma/v(1)^2); 
    v = v / v(1) ;
end
   
