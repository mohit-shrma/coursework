%find optimal policy by policy iteration for a reward function
% returns optimal policy for each state {u,d,l,r}
% and outputs row, column, policy for each non-terminal state
%TODO: output row, col and policy

function [utilities, policy] = mdpPI(reward)
%{
   CSci5512 Spring'12 Homework 3
   login: sharm163@umn.edu
   date: 4/11/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: mdpPI
%}

%3 X 4
worldSize = [3 4];

%states in world
% 1 -> valid state
% 2 ->  + terminal state
% -2 ->  + terminal state
% -1 -> invalid state/wall
states = [1 1 1 2;
          1 -1 1 -2;
          1 1 1 1];

%initialize state utilities
utilities = zeros(size(states));
utilities(1, 4) = 1;
utilities(2, 4) = -1;


%initialize discount factor
%TODO: choose appropriate discount factor
discFactor = 0.99;

%moves Type
movesType = ['u' 'd' 'l' 'r'];

%movesName
moveUp = 'u';
moveDown = 'd';
moveLeft = 'l';
moveRight = 'r';

%initialize state policies at random
% valid values 'u', 'd', 'l', 'r
policy = zeros(size(states));
for rowIter=1:size(states,1)
    for colIter=1:size(states,2)
         %if state is terminal or invalid then continue
         if states(rowIter, colIter) ~= 1
             continue;
         end
         %pick a random move from available moves
         %movIndx = randi(length(movesType));
         %policy(rowIter, colIter) = movesType(movIndx);
         policy(rowIter, colIter) = moveLeft;
    end
end

%Transition prob for moves
%in order 'forward', 'side-left', 'side-right', 'down'
%denoted by 0, 1, 2, 3
transition = [0.8 0.1 0.1 0];

probFwd = transition(1);
probSideLeft = transition(2);
probSideRight = transition(3);
probBack = transition(4);

changed = true;

while changed
    
    utilities = policyEvaluation(policy, states, transition, reward, discFactor);
    changed = false;
    
    for rowIter=1:size(states,1)
        for colIter=1:size(states,2)
            
            % state is given by pair (rowIter, colIter)
            
            %if state is terminal or invalid then continue
            if states(rowIter, colIter) ~= 1
                continue;
            end
            
            %get all possible states that can be reached from
            %current state
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
            utilityUp = utilities(nextStateRow, nextStateCol);
            
            
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
            utilityLeft = utilities(nextStateRow, nextStateCol);
            
            
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
            utilityRight = utilities(nextStateRow, nextStateCol);
            
        
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
            utilityDown = utilities(nextStateRow, nextStateCol);
            
            %policy value for all possible moves
            possiblePolicyVals = zeros(size(movesType));
            
            %find max policy mov, and value
            maxPolicyMov = '';
            maxPolicyVal = -9999;
            
            %for each move, find the move that give maximum gain
            for iter=1:length(movesType)
                
                if movesType(iter) == moveUp
                    movVal = probFwd*utilityUp + ...
                             probSideLeft*utilityLeft + probSideRight*utilityRight ...
                             + probBack*utilityDown;
                elseif movesType(iter) == moveDown
                    movVal = probBack*utilityUp + ...
                             probSideRight*utilityLeft + probSideLeft*utilityRight ...
                             + probFwd*utilityDown;
                elseif movesType(iter) == moveLeft
                    movVal = probSideRight*utilityUp + ...
                             probFwd*utilityLeft + probBack*utilityRight + ...
                             probSideLeft*utilityDown;;
                elseif movesType(iter) == moveRight
                    movVal = probSideLeft*utilityUp + ...
                             probBack*utilityLeft + probFwd*utilityRight + ...
                             probSideRight*utilityDown;;
                end
                
                possiblePolicyVals(iter) = movVal;
                
                if movVal > maxPolicyVal
                    maxPolicyVal = movVal;
                    maxPolicyMov = movesType(iter);
                end
                
            end
            
            %if the maximum mov is different from our current move
            %in policy, improve our currently  evaluated policy
            if maxPolicyMov ~= policy(rowIter,colIter)
                policy(rowIter, colIter) = maxPolicyMov;
                changed = true;
            end
            
            
        end % end col iteration
    end %end row iteration
end % end while changed
%display the policy found by policy iteration
fprintf('\n policy by policy iteration and reward = %f \n', reward);
char(policy)
