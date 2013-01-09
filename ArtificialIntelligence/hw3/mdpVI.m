%find optimal policy by value iteration for a reward function
% returns optimal policy for each state {u,d,l,r}
% and outputs row, column, policy for each non-terminal state

function [utilities, policy] = mdpVI(reward)
%{
   CSci5512 Spring'12 Homework 3
   login: sharm163@umn.edu
   date: 4/11/2012
   name: Mohit Sharma
   id: 4465482
   algorithm: mdpVI
%}

%3 X 4
worldSize = [3 4];

%states in world
% 1 -> valid state
% 0 -> terminal state
% -1 -> invalid state/wall
states = [1 1 1 0;
          1 -1 1 0;
          1 1 1 1];

%initialize state utilities
utilities = zeros(size(states));
utilities(1, 4) = 1;
utilities(2, 4) = -1;

%initialize state policies
% valid values 'u', 'd', 'l', 'r
policy = zeros(size(states));

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

%Transition prob for moves
%in order 'forward', 'side-left', 'side-right', 'down'
%denoted by 0, 1, 2, 3
transition = [0.8 0.1 0.1 0];

probFwd = transition(1);
probSideLeft = transition(2);
probSideRight = transition(3);
probBack = transition(4);

%TODO: choose an appropriate e, error allowed in utility of state
%initialize error allowed in utility of state
e = 0.00001;

%max change in utility of any state in iteration
delta  = 0;
iter = 0;
while true
    iter = iter +1;
    delta = 0;
    %for each state in world
    for rowIter=1:size(states, 1)
        for colIter=1:size(states,2)
            % state is given by pair (rowIter, colIter)
        
            %if state is teminal or invalid then continue
            if states(rowIter, colIter) ~= 1
                continue;
            end
        
            %for each possible move compute the final utility and
            %update the policy with the move which gives maximum
            %utility
            tempMaxMove = movesType(1);
            %TODO: consider large -ve values too
            tempMaxMoveSum = -999999;
        
            %get all possible states utilities around current state 
            
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
            
        
            for moveIter=1:length(movesType)
            
                weightedUtilSum = 0;         
            
                if movesType(moveIter) == moveUp
                    %move up
                            
                    %if actually up
                    weightedUtilSum = weightedUtilSum + probFwd*utilityUp;
                                
                    %if actually left
                    weightedUtilSum = weightedUtilSum + probSideLeft*utilityLeft;
                
                    %if actually right
                    weightedUtilSum = weightedUtilSum + probSideRight*utilityRight;
                
                    %if actually down
                    weightedUtilSum = weightedUtilSum + probBack*utilityDown;
                
                elseif movesType(moveIter) == moveDown
                    %move down
                                        
                    %if actually up
                    weightedUtilSum = weightedUtilSum + probBack*utilityUp;
                                
                    %if actually left
                    weightedUtilSum = weightedUtilSum + probSideRight*utilityLeft;
                
                    %if actually right
                    weightedUtilSum = weightedUtilSum + probSideLeft*utilityRight;
                
                    %if actually down
                    weightedUtilSum = weightedUtilSum + probFwd*utilityDown;
            
                            
                elseif movesType(moveIter) == moveLeft
                    %move left
                
                    %if actually up
                    weightedUtilSum = weightedUtilSum + probSideRight*utilityUp;
                                
                    %if actually left
                    weightedUtilSum = weightedUtilSum + probFwd*utilityLeft;
                
                    %if actually right
                    weightedUtilSum = weightedUtilSum + probBack*utilityRight;
                
                    %if actually down
                    weightedUtilSum = weightedUtilSum + probSideLeft*utilityDown;
            
                
                elseif movesType(moveIter) == moveRight
                    %move right
                
                    %if actually up
                    weightedUtilSum = weightedUtilSum + probSideLeft*utilityUp;
                                
                    %if actually left
                    weightedUtilSum = weightedUtilSum + probBack*utilityLeft;
                
                    %if actually right
                    weightedUtilSum = weightedUtilSum + probFwd*utilityRight;
                
                    %if actually down
                    weightedUtilSum = weightedUtilSum + probSideRight*utilityDown;

                end
            
                %now check if existing max sum lesser for this move
                if tempMaxMoveSum < weightedUtilSum
                    tempMaxMoveSum = weightedUtilSum;
                    tempMaxMove = movesType(moveIter);
                end
            end
            
            %end of iteration of all moves
            
            %update current state utility
            oldUtility = utilities(rowIter, colIter);
            utilities(rowIter, colIter) = reward + discFactor* ...
                tempMaxMoveSum;
            policy(rowIter, colIter) = tempMaxMove;
            
            diff = abs(utilities(rowIter, colIter) - oldUtility);
            
            if diff > delta
                delta = diff;
            end
            
        end %end column iteration
    end %end row iteration
        
    % repeat above until delta is less than some threshold
    if delta < ((e*(1 - discFactor))/discFactor)
        break;
    end
        
end
%display policy found by value iteration
fprintf('\n policy by policy iteration and reward = %f \n', reward);
char(policy)

