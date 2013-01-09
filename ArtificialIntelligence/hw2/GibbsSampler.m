%{
   CSci5512 Spring'12 Homework 2
   login: sharm163@umn.edu
   date: 3/4/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: GibbsSampler
%}

function X = GibbsSampler(numIter, cptRainyGivenCloudyNS_T_W_T, ...
                            cptCloudyGivenRainyNS_T_W_T)
%instantiate X(1) = [ R C ]
X = [2 2];

for iter = 2:numIter
    %pick variable at random
    choice = unidrnd(2);
    if choice == 1
        %selected rain
        %sample rain from cptRainyGivenCloudy
        rand_p = unifrnd(0,1);
        if rand_p <= cptRainyGivenCloudyNS_T_W_T( X(iter-1, 2) )
            %set rainy true
            X = [X; 2 X(iter-1, 2)];
        else
            %set rainy false
            X = [X; 1 X(iter-1, 2)];
        end
    else
        %selected cloudy
        %sample cloudy from cptCloudyGivenRainy
        rand_p = unifrnd(0,1);
        if rand_p <= cptCloudyGivenRainyNS_T_W_T( X(iter-1, 1) )
            %set cloudy true
            X = [X; X(iter-1, 1) 2];
        else
            %set cloudy false
            X = [X; X(iter-1, 1) 1];
        end
    end
end
