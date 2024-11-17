function [newAgent,trainStats] = GliderTrain(agent,env)
% [NEWAGENT,TRAINSTATS] = mytrain(AGENT,ENV) train AGENT within ENVIRONMENT
% with the training options specified on the Train tab of the Reinforcement Learning Designer app.
% mytrain returns trained agent NEWAGENT and training statistics TRAINSTATS.

% Reinforcement Learning Toolbox
% Generated on: 10-Jan-2024 18:38:33

%% Create training options
trainOptions = rlTrainingOptions();
trainOptions.MaxEpisodes = 800;
trainOptions.MaxStepsPerEpisode = 3000;
trainOptions.ScoreAveragingWindowLength = 500;
trainOptions.StopTrainingCriteria = "AverageReward";
trainOptions.StopTrainingValue = 0;

%% Make copy of agent
newAgent = copy(agent);
% resetting the initial epsilon!
% newAgent.AgentOptions.EpsilonGreedyExploration.Epsilon = 0.5;

%% Perform training
trainStats = train(newAgent,env,trainOptions);
end