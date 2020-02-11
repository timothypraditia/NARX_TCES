close all
clear all

%% Main script to run PINN for TCES

%  Timothy Praditia
%  Thilo Walser
%  Sergey Oladyshkin
%  Wolfgang Nowak

%  Department of Stochastic Simulation and Safety Research for Hydrosystems
%  Institute for Modelling Hydraulic and Environmental Systems
%  University of Stuttgart

%  2019

%% Adding modified training script to MATLAB search path

% Path might need to be adjusted
addpath(genpath('c:/Program Files/MATLAB/2019a/toolbox/nnet'));

%% Hyperparameters
% Depending on whether you modified the 'trainbr' script directly or make a
% copy of the script, the trainingFcnString has to be adjusted accordingly
% into the modified file name
trainingFcnString = 'trainbr';

hiddenLayerSize = [15,8];           %Sizes of hidden layers (add column for new layer, increase number for more neurons in layer) 
feedback_delay = 5;                 %number of previous time steps to be taken for target calculation
Ntrial = 50;                        %Number of reinitializations + retrainings per configuration
train_series = 100;                 %Number of time series used for training (between 1 - 2360)

%% Load data to workspace

val_series = round(train_series/5);
load('io_data')         %unprocessed input (not noisy) and contains n_out and Q_out
load('noisy_data')      %noisy training data for 4 target variables 
Y_start = Y_train{1,feedback_delay}(:,1:train_series); %time step t = 0

for t = 1:size(Y_train_noisy,2)
    n_out{1,t} = n_out_train{1,t}(1:train_series);
    Q_out{1,t} = Q_out_train{1,t}(1:train_series);
    Y_train_noisy{1,t} = Y_train_noisy{1,t}(1:4,1:train_series);
    X_train{1,t} = X_train{1,t}(:,1:train_series);
    Y_val_noisy{1,t} = Y_val_noisy{1,t}(1:4,1:val_series);
    X_val{1,t} = X_val{1,t}(:,1:val_series);
    Y_test{1,t} = Y_test{1,t}(1:4,:);
end

fprintf(strcat(['data containing ',num2str(size(X_train{1,1},2)),' training series and ',num2str(size(X_test{1,1},2)),' test series\n']))
fprintf(strcat([num2str(hiddenLayerSize),' hidden layers, ',num2str(feedback_delay),' delay time steps, ',num2str(Ntrial),' = Ntrial\n']))

% Creating result folder
folder_o = strcat('results/',date,'noisy_',trainingFcnString,'_layers_',num2str(hiddenLayerSize),'_delay_',num2str(feedback_delay),'_training_',num2str(train_series));  
mkdir(folder_o)

perf_best = 1000;

%% Start training with different initializations

for n = 1:Ntrial
    fprintf(strcat(['Start of Trial number ',num2str(n),'\n']))
    
    %% Create a Nonlinear Autoregressive Network with External Input
    inputDelays = 1:1;                               %time delay of exogeneous inputs for NN
    feedbackDelays = 1:feedback_delay;               %time delay of targets (outputs) for NN

    % Training-Function: Levenberg-Marquardt backpropagation in closed loop
    net_temp = narxnet(inputDelays,feedbackDelays,hiddenLayerSize,'closed',trainingFcnString); 

    %% Setting network parameters
    % Setup Division of Data for Training, Validation, Testing and
    % additional training parameters
    net_temp.divideFcn = 'dividetrain';                         %all training data used for training
    net_temp.performParam.normalization = 'standard';           %normalization applied
    net_temp.trainParam.epochs = 500;                           %maximum number of iterations (epochs)
    net_temp.trainParam.mu_max = 1E10;                          %maximum mu value
    net_temp.trainParam.n_out = n_out(1,feedback_delay+1:1000); %outflowing H2O mole
    net_temp.trainParam.Q_out = Q_out(1,feedback_delay+1:1000); %outflowing energy
    net_temp.trainParam.Z_MB = 3.993615e+03;                    %spread of mole balance error
    net_temp.trainParam.Z_EB = 318295000;                       %spread of energy balance error

    net_temp.trainParam.exInputs = X_train{1,1};                %exogeneous inputs, needed for modified training function
    net_temp.trainParam.startValues = Y_start;                  %target start values (of 0th time step)
    net_temp.trainParam.Q_in = Q_in_train(1,1:train_series);    %inflowing energy

    %% Preparing timeseries data
    
    %Xs - cell containing inputs X and targets T of delayed data (if time delay is 5 the first entry will be the one of the 6th time step)
    %Xi - initial input delay states
    %Ai - initial layer delay states
    %Ts - cell containing only shifted targets T of delayed data 
    
    [Xs,Xi,Ai,Ts] = preparets(net_temp,X_train,{},Y_train_noisy);

    %% Train the Network
    
    [net_temp,tr_all{n}] = train(net_temp,Xs,Ts,Xi,Ai);
    y_train = net_temp(Xs,Xi,Ai);

    % close network 
    netc_temp = closeloop(net_temp);

    %% Calculate the performance of the network on validation examples in closed loop
    [Xc,Xic,Aic,Tc] = preparets(netc_temp,X_val,{},Y_val_noisy);
    y_NARX_val = netc_temp(Xc,Xic,Aic);
    perf_val(n) = perform(netc_temp,Tc,y_NARX_val);
    net_all{n} = netc_temp;

    if perf_val(n) < perf_best
        perf_best = perf_val(n);
        netc = netc_temp;
        tr = tr_all{n};
    end
    
    %% calculate the performance of the network on test examples in closed loop
    [Xt,Xit,Ait,Tt] = preparets(netc_temp,X_test,{},Y_test);
    y_NARX_test = netc_temp(Xt,Xit,Ait);
    perf_test(n) = perform(netc_temp,Tc,y_NARX_test);
    
    fprintf(strcat(['Validation error: ',num2str(perf_val(n)),'\n']))
    fprintf(strcat(['Test error: ',num2str(perf_test(n)),'\n\n']))
end

%% Save results

save(strcat(folder_o,'/network.mat'),'netc','net_all','tr','tr_all','perf_val','perf_test')

%% Plot results

plot_best_worst(netc,X_test,Y_test,feedback_delay)
