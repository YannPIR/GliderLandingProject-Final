clear
clc
%% step 1: creating the enviroment envA1

ObsInfo = rlNumericSpec([4 1]);
ObsInfo.Name = "Glider States";
ObsInfo.Description = 'x, y, v, theta';%, ind, wch

ActInfo = rlFiniteSetSpec([1 4 7]);
ActInfo.Name = "Glider Action";

envA1 = rlFunctionEnv(ObsInfo,ActInfo,"GliderFinalStepFunction","GliderResetFunction");

%% step 2: creating the agent

% agent
obsInfo = getObservationInfo(envA1);
actInfo = getActionInfo(envA1);
%rng(0) :if we want reproductivity

dnn = [
    featureInputLayer(prod(obsInfo.Dimension))
    fullyConnectedLayer(32)
    reluLayer
    %fullyConnectedLayer(32)% added
    %reluLayer
    fullyConnectedLayer(32)
    reluLayer
    fullyConnectedLayer(numel(actInfo.Elements))
    ];

dnn = dlnetwork(dnn);
summary(dnn)

% Plot network
plot(dnn)

critic = rlVectorQValueFunction(dnn,obsInfo,actInfo);
%getValue(critic,{rand(obsInfo.Dimension)})%
criticOptions = rlOptimizerOptions( ...
    LearnRate=0.02);
    % GradientThreshold=Inf,...
    % Algorithm="adam",...
    % GradientThresholdMethod="l2norm",...
    % L2RegularizationFactor=0.0001
    %,...OptimizerParameters=!?
    
agentOptions = rlDQNAgentOptions(...
    SampleTime=1, ... %BatchDataRegularizerOptions=x,...
    CriticOptimizerOptions=criticOptions,...
    DiscountFactor=0.99,...
    ExperienceBufferLength=1e+04,... %InfoToSave=x,...
    MiniBatchSize=64,... %NumStepsToLookAhead=x,...%ResetExperienceBufferBeforeTraining=x,...%SequenceLength=32,...
    TargetSmoothFactor=0.001,...
    TargetUpdateFrequency=1,...
    UseDoubleDQN=true);

%agentOptions

agent = rlDQNAgent(critic,agentOptions);
agent.AgentOptions.EpsilonGreedyExploration.Epsilon = 1;
agent.AgentOptions.EpsilonGreedyExploration.EpsilonDecay = 0.005;
agent.AgentOptions.EpsilonGreedyExploration.EpsilonMin = 0.01;

%getAction(agent,rand(obsInfo.Dimension))%
%% step 3: training

[TrainedGlider,trainStats] = GliderTrain(agent,envA1);
% saving the agent:
save TrainedGlider.mat TrainedGlider

