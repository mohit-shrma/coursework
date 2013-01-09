function [x] = MaxSeq(n, inputEvid)
%{
   CSci5512 Spring'12 Homework 2
   login: sharm163@umn.edu
   date: 3/4/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: MaxSeq
%}

%{
  implementation for maximal sequence
  e.g : MaxSeq(10, [0 1 0 1 0 1 0 1 0 1 ])
%}

%initialize return vector
x = [];

if n ~= length(inputEvid)
    fprintf(['input evidence array dont agree with the length ' ...
             'specified']);
else
    %initial probabilities/ prior
    X0 = [0.5 ; 0.5];

    %transition matrix p(Xt|Xt-1)
    T = [0.7 0.3; 
         0.4 .6];
     
    %transition matrix for top-left being P(Ft|Ft-1) and bottom-right
    %P(Tt|Tt-1)
    Tmirror = [0.6 0.4;
               0.3 0.7];
           
    %evidence matrix p(Et|Xt)
    E = [0.9 0.1; 
         0.3 0.7];
     
    %evidence matrix top-left being P(Ef|Xf) and bottom-right P(Et|Xt)
    Emirror = [0.7 0.3;
               0.1 0.9];
    
    %evidence matrix to use given evidence is true 
    evidenceTrue = [0.9 0;
                    0 0.3];

    evidenceFalse = [0.1 0;
                     0 0.7];
                 
    %using 1 for F and 2 for T    
    %that's why increment our each element in input vector by 1
    inputEvid = inputEvid + 1;

    lengthOfEvidence = length(inputEvid);

    %inputEvid1 = [ 1 1 1 2 2 2 2 1 1 1];
    %inputEvid2 = [ 1 2 1 2 1 2 1 2 1 2];

    %create a num_states/evidences X possible_states_val X 2  matrix to store for
    %each possible value of a state -> M1:t, backward
    %pointer to prev state value or simply prev state value
    %call this matrix dpMat
    %dpMat(i, 1, :) -> info when value = First possible state value
    %e.g. False; dpMat(i, 2, :) -> value when e.g. True
    %dpMat(i, 1, 1) -> M1:i 
    %dpMat(i, 1, 2) -> i-1 state value 
    dpMat = zeros(lengthOfEvidence, size(X0, 1), 2);  

    for iter=1:lengthOfEvidence
        %do computation for state iter
        %compute dpMat(iter, :, :)

        %tempMaxState = -1;
        %tempMaxVal = -1;
        for state=1:size(X0,1)
            %compute dpMat(iter, state, :) for the current state

            tempMaxPrevState = -1;
            % max <Xt> ( p(Xt+1|Xt) m1:t )
            tempMaxPrevVal = -1;
            %for each previous state's(iter-1) m1:(iter-1)
            for prevState=1:size(X0,1)
                %do computation based on dpMat(iter-1, prevState, 1)
                if iter-1 == 0
                    m_1_prevState = X0(prevState);
                else
                    m_1_prevState = dpMat(iter-1, prevState, 1);
                end

                % p(state|prevState)
                tempVal = Tmirror(prevState, state)  * m_1_prevState;
                if tempMaxPrevVal < tempVal
                    tempMaxPrevVal = tempVal;
                    tempMaxPrevState = prevState;
                end
            end

            %compute p(Et+1|Xt+1) * tempMaxPrevVal
            tempMaxVal = Emirror(state, inputEvid(iter)) * tempMaxPrevVal;
            dpMat(iter, state, :) = [tempMaxVal; tempMaxPrevState];
        end

        %need to normalize for given possible states all m1:t
        dpMat(iter, :, 1) = dpMat(iter, :, 1)/sum(dpMat(iter, :, 1));
    end

    %get last node's state having largest value
    maxLastStateVal = -1;
    maxLastState = -1;
    for state=1:size(X0,1)
        if dpMat(lengthOfEvidence, state,1) > maxLastStateVal
            maxLastStateVal = dpMat(lengthOfEvidence, state,1);
            maxLastState = state;
        end    
    end

    %initialize the last backward state to max computed above
    bwdState = maxLastState;
    %fprintf('\t %d', bwdState);
    
    %preinitialize output vector
    x = zeros(1, lengthOfEvidence);
    
    %backtrace from the last state
    for iter=lengthOfEvidence : -1 : 1
        x(iter) = bwdState;
        bwdState = dpMat(iter, bwdState, 2);
        %fprintf('\t %d', bwdState);    
    end
    
     x = x - 1;
     x
end

exit;