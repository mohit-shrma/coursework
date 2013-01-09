%implements filtering using forward propagation

function [trueProb] = tpUmbrella(evidence)
%{
   CSci5512 Spring'12 Homework 3
   login: sharm163@umn.edu
   date: 4/11/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: tpUmbrella
%}

%initial probabilities/ prior
%1->T, 2->F
X0 = [0.5 ;
      0.5];

%transition matrix p(Xt|Xt-1)
%1,1 -> p(Xt=T|Xt-1=T) 
T = [0.7 0.3; 
     0.3 0.7];

%evidence matrix p(Et|Xt)
%1,1 -> p(Et=T|Xt=T)
E = [0.9 0.1; 
     0.2 0.8];

%evidence matrix to use given evidence is true
%1,1 denotes Et=T|Rt=T 2,2 -> Et=T|Rt=F 
evidenceTrue = [0.9 0;
                0 0.2];

%1,1 denotes Et=F|Rt=T 2,2 -> Et=F|Rt=F 
evidenceFalse = [0.1 0;
                 0 0.8];

%initialize fwdT to prior
fwdT = X0;
for iter=1:length(evidence)
    %initialize given evidence matrix to state when evidence given
    %is true
    evidenceMat = evidenceTrue;
    
    if evidence(iter) == 0
        %given evidence is false, change evidenceMat
        evidenceMat = evidenceFalse;
    end
    
    fwdT = evidenceMat * T' * fwdT;
    fwdT = fwdT / sum(fwdT);
end

%return flipped value as it expects 'F' first then 'T'
trueProb = [fwdT(2);fwdT(1)];

fprintf('\n true prob. by exact forward filtering, P(R=F) => %f\n',...
            trueProb(1));
 
fprintf('\n true prob. by exact forward filtering, P(R=T) => %f\n',...
            trueProb(2));