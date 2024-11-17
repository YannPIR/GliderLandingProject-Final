%% step 1: creating the test enviroment

%rng(2); % choose another number for different initial condition

ObsInfo = rlNumericSpec([4 1]);
ObsInfo.Name = "Glider States";
ObsInfo.Description = 'x, y, v, theta';

ActInfo = rlFiniteSetSpec([1 4 7]);
ActInfo.Name = "Glider Action";

testenv = rlFunctionEnv(ObsInfo,ActInfo,"GliderFinalStepFunction","GliderResetFunction");
%testenv2 = rlFunctionEnv(ObsInfo,ActInfo,"GliderOtherStepFunction","GliderResetFunction");

%% step 2: loading an existing agent and running a simulation

stepnumber=3000;
simOptions = rlSimulationOptions(MaxSteps=stepnumber);

load TrainedGlider.mat
experience2= sim(testenv,TrainedGlider,simOptions);
totalReward2 = sum(experience2.Reward);

load FinalGlider.mat
experience1 = sim(testenv,Glider3,simOptions);
totalReward1 = sum(experience1.Reward);

%% step 3: reference solution with fixed wings

refsol = zeros(4,stepnumber+1); % MaxSteps=10000
refsol(:,1) = experience1.Observation.GliderStates.Data(1:4,1);

h = 0.05;
for j=1:stepnumber
    refsol(:,j+1) =RK4(h,refsol(:,j)); %RK4 for fixed wing state
end

%% step 4: plots

figure
%plot(refsol(1,:),refsol(2,:),'--r')
%hold on
hold on
plot(experience1.Observation.GliderStates.Data(1,:),experience1.Observation.GliderStates.Data(2,:),'b')
hold on
plot(experience2.Observation.GliderStates.Data(1,:),experience2.Observation.GliderStates.Data(2,:),'g')
yline(0,'r')
yline(10,'--b')
xlabel('x')
ylabel('y')
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------------
function NextState = RK4(h,State)

    Y1=State;
    k1=Glide(Y1);
    Y2=State+h/2*k1;
    k2=Glide(Y2);
    Y3=State+h/2*k2;
    k3=Glide(Y3);
    Y4=State+h*k3;
    k4=Glide(Y4);
    NextState=State+h*(k1+2*k2+2*k3+k4)/6;
end
%----------------------------------
function f = Glide(y)
    g=9.81;
    muD=1.0848*10^(-4); %Drag coeff 2
    muL=0.0077; %Lift coeff 2

	f = zeros(4,1);
	f(1) = y(3)*cos(y(4));
	f(2) = y(3)*sin(y(4));
	f(3) = -g*sin(y(4))-muD*y(3)^2;
    f(4) = -(g/y(3))*cos(y(4))+muL*y(3);
end