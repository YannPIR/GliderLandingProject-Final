function [InitialObservation, InitialState] = GliderResetFunction()
% Reset function to place custom glider environment into a random
% initial state.

%rng(3);%86 76 106

% X
X0 = 0;
% Y rand
Y0 = 70 + randi([-1,2])*10;
% V rand
V0 = 10 + (rand-1/2)*10;
% theta rand
theta0 = 0;% randi([-9,1])*pi/36; % [-45;5]°

% Return initial environment state variables as logged signals.
InitialState = [X0;Y0;V0;theta0];
InitialObservation = InitialState;
end