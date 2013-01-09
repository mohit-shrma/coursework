%solve for utility values for gven policy
function [utilities] = policyEvaluation(policy, states, transition, ...
                                        reward, discFactor)
%{
   CSci5512 Spring'12 Homework 3
   login: sharm163@umn.edu
   date: 4/11/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: policyEvaluation
%}

%moves Type
movesType = ['u' 'd' 'l' 'r'];

%movesName
moveUp = 'u';
moveDown = 'd';
moveLeft = 'l';
moveRight = 'r';

%initialize state utilities
utilities = zeros(size(states));
utilities(1, 4) = 1;
utilities(2, 4) = -1;

probFwd = transition(1);
probSideLeft = transition(2);
probSideRight = transition(3);
probBack = transition(4);


worldSize = size(states, 1) * size(states, 2);

variablesWeightageVec = [];
posTerminalStatesIndice = [];
negTerminalStatesIndice = [];
wallStatesIndice = [];

% for each state assign weightage in variables vec
% state index in vec => (rowindice-1)*cols + colindice

for rowIter=1:size(states, 1)
    for colIter=1:size(states,2)
        
        %temporary state vars vec
        tempVars =  zeros(1, worldSize);
        
        %continue only if valid state
        %if state is terminal or invalid then continue
        if states(rowIter, colIter) ~= 1
            variablesWeightageVec = [variablesWeightageVec; ...
                                tempVars];
            if states(rowIter, colIter) == 2
                posTerminalStatesIndice = [posTerminalStatesIndice; ...
                   (rowIter-1)*size(states,2) + colIter];
            elseif states(rowIter, colIter) == -2
                negTerminalStatesIndice = [negTerminalStatesIndice; ...
                   (rowIter-1)*size(states,2) + colIter];
            elseif states(rowIter, colIter) == -1
                wallStatesIndice = [wallStatesIndice;... 
                                    ((rowIter-1)* size(states,2)) + ...
                                    colIter];
            end
            
            continue;
        end
        
        %get indices of nearby states
        %upState
        if rowIter == 1
            nextStateRow = 1;
        else
            nextStateRow = rowIter - 1;
        end
        nextStateCol = colIter;
        %if next state invalid then don't move
        if states(nextStateRow, nextStateCol) == -1
            nextStateRow = rowIter;
            nextStateCol = colIter;
        end
        upStateIndice = size(states, 2)*(nextStateRow-1) + ...
            nextStateCol;
        
        %leftState
        if colIter == 1
            nextStateCol = 1; 
        else
            nextStateCol = colIter - 1;
        end
        nextStateRow = rowIter;
        %if next state invalid then don't move
        if states(nextStateRow, nextStateCol) == -1
            nextStateRow = rowIter;
            nextStateCol = colIter;
        end        
        leftStateIndice = size(states, 2)*(nextStateRow-1) + ...
            nextStateCol;
        
        
        %rightState
        if mod(colIter,size(states, 2)) == 0
            nextStateCol = colIter; 
        else
            nextStateCol = colIter + 1;
        end
        nextStateRow = rowIter;
        %if next state invalid then don't move
        if states(nextStateRow, nextStateCol) == -1
            nextStateRow = rowIter;
            nextStateCol = colIter;
        end        
        rightStateIndice = size(states, 2)*(nextStateRow-1) + ...
            nextStateCol;
        
        
        %downState
        if mod(rowIter,size(states, 1)) == 0
            nextStateRow = rowIter; 
        else
            nextStateRow = rowIter + 1;
        end
        nextStateCol = colIter;
        %if next state invalid then don't move
        if states(nextStateRow, nextStateCol) == -1
            nextStateRow = rowIter;
            nextStateCol = colIter;
        end        
        downStateIndice = size(states, 2)*(nextStateRow-1) + ...
            nextStateCol;
        
        
        
        %depending on the policy make linear equation in terms of
        %utilities of nearby states
        currPolicy = policy(rowIter, colIter);
        
        if currPolicy == moveUp
            tempVars(upStateIndice) = tempVars(upStateIndice) + probFwd;
            tempVars(leftStateIndice) = tempVars(leftStateIndice) +probSideLeft;
            tempVars(rightStateIndice) = tempVars(rightStateIndice) + probSideRight;
            tempVars(downStateIndice) = tempVars(downStateIndice) + probBack;
        elseif currPolicy == moveDown
            tempVars(upStateIndice) = tempVars(upStateIndice) + probBack;
            tempVars(leftStateIndice) = tempVars(leftStateIndice) + probSideRight;
            tempVars(rightStateIndice) = tempVars(rightStateIndice) + probSideLeft;
            tempVars(downStateIndice) = tempVars(downStateIndice) + probFwd;            
        elseif currPolicy == moveLeft
            tempVars(upStateIndice) = tempVars(upStateIndice) + probSideRight;
            tempVars(leftStateIndice) = tempVars(leftStateIndice) + probFwd;
            tempVars(rightStateIndice) = tempVars(rightStateIndice) + probBack;
            tempVars(downStateIndice) = tempVars(downStateIndice) + probSideLeft;                        
        elseif currPolicy == moveRight
            tempVars(upStateIndice) = tempVars(upStateIndice) + probSideLeft;
            tempVars(leftStateIndice) = tempVars(leftStateIndice) + probBack;
            tempVars(rightStateIndice) = tempVars(rightStateIndice) + probFwd;
            tempVars(downStateIndice) = tempVars(downStateIndice) + probSideRight;
        end
        
        variablesWeightageVec = [variablesWeightageVec; tempVars];
        
    end % end colIter
end %end rowIter

variablesWeightageVec = variablesWeightageVec.*discFactor;

%in AX = B, this is our B
rewardsVec = ones(worldSize, 1);
rewardsVec = rewardsVec.*(-1*reward);


posTerminalStateInd = posTerminalStatesIndice(1);
negTerminalStateInd = negTerminalStatesIndice(1);
wallStateInd = wallStatesIndice(1);


%for each row in variables weightage vec, sanitize it that is
%remove wall, apply terminal states to reward, and if same state
%invloved the subtract -1

for iter=1:size(variablesWeightageVec,1)
    
    %subtract -1 in case of same state
    tempVars = variablesWeightageVec(iter, :);
    tempVars(iter) = tempVars(iter) - 1;
    variablesWeightageVec(iter, :) = tempVars; 
    
    %find terminal states and apply to reward
    rewardsVec(iter) = rewardsVec(iter) - (1*tempVars(posTerminalStateInd));
   
    rewardsVec(iter) = rewardsVec(iter) - (-1*tempVars(negTerminalStateInd));
    
%     for posTermIter=1:size(posTerminalStatesIndice, 1)
%         tempTermInd = posTerminalStatesIndice(posTermIter);
%         rewardsVec(iter) = rewardsVec(iter) - ...
%             (1*tempVars(tempTermInd));  
%     end
%     
%     for negTermIter=1:size(negTerminalStatesIndice, 1)
%         tempTermInd = negTerminalStatesIndice(negTermIter);
%         rewardsVec(iter) = rewardsVec(iter) - ...
%             (-1*tempVars(tempTermInd));  
%     end
    
end


%remove terminal state column from variables weightage vec

actualVariables = [];
for rowIter=1:size(variablesWeightageVec, 1)
    
    if rowIter == posTerminalStateInd || rowIter == negTerminalStateInd ...
                || rowIter == wallStateInd
        continue;
    end
    
    tempRow = [];
    for colIter=1:size(variablesWeightageVec, 2)
        if colIter == posTerminalStateInd || colIter == negTerminalStateInd ...
                 || colIter == wallStateInd
            continue;
        end

        tempRow = [tempRow variablesWeightageVec(rowIter, colIter)];
    end
    actualVariables = [actualVariables; tempRow];
end

actualRewardsVec = [];
for iter = 1:size(rewardsVec,1)
    if iter == posTerminalStateInd || iter == negTerminalStateInd ...
                || iter == wallStateInd
        continue;
    end
    actualRewardsVec = [actualRewardsVec; rewardsVec(iter)];
end


%solve linear equation
tempUtilities = linsolve(actualVariables, actualRewardsVec);
counter = 1;

utilities = zeros(size(states));

for rowIter=1:size(states, 1)
    for colIter=1:size(states,2)
        
        %if terminal then assign +1/-1
        if states(rowIter, colIter) == 2
            utilities(rowIter, colIter) = 1;
            continue;
        end
        
        if states(rowIter, colIter) == -2
            utilities(rowIter, colIter) = -1;
            continue;
        end
        
        if states(rowIter, colIter) == -1
            utilities(rowIter, colIter) = 0;
            continue;
        end
        
        utilities(rowIter, colIter) = tempUtilities(counter);
        counter = counter + 1;
    end
end
