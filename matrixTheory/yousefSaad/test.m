%% data fitting test 
load('dat');
%
tt = dat(:,1); 
yy = dat(:,2);
n = size(dat,1);
%
%%fn1 = inline('t .* cos(3*pi*t)') ;
fn1 = inline('cos(3*pi*t)') ;
fn2 = inline('t.^2');
%%-------------------- get F matrix 
f1 = ones(n,1) ;
f2 = tt;
f3 = fn1(tt); 
f4 = fn2(tt);
F = [f1 f2 f3 f4] ;
%%-------------------- compute QR fact.  
[Q, R]  = mgsa(F);
%%-------------------- solve 
coef = R \ (Q'*yy);
zz = F*coef;
%%-------------------- plot
pts = linspace(min(tt),max(tt),100);
pts = pts';
yp  = coef(1)*ones(100,1)+coef(2)*pts ...
    + coef(3) * fn1(pts)+coef(4)*fn2(pts);

plot(tt,zz,'k*','linewidth',2) ;
hold on 
plot(pts,yp,'b-') 
plot(tt,yy,'ro','linewidth',2) 
plot(tt,yy,'k--') 
