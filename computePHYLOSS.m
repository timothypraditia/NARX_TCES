%% Function to calculate physical error for loss function calculation

%  Timothy Praditia
%  Thilo Walser
%  Sergey Oladyshkin
%  Wolfgang Nowak

%  Department of Stochastic Simulation and Safety Research for Hydrosystems
%  Institute for Modelling Hydraulic and Environmental Systems
%  University of Stuttgart

%  2019

%% NOTE

% Copy the function and put it in the modified 'trainbr.m' file inside the
% nnet Toolbox: MATLAB>'version'>toolbox>nnet>nnet>nntrain

% Please note that inside the 'trainbr.m' file, the functions:
% initializeTraining, trainingIteration, and computeDX also has to be
% modified accordingly to initialize the hyperparameters, redefine the loss
% function, update hyperparameters, and update network weights and biases

%% Function Start

% This function returns:
% sum_PHYLOSS: an array containing the mean squared physical error
% PHYLOSS: a cell array containing the individual physical error at
% different time steps and samples

% Even though the function 'computeJPHY' also returns sum_PHYLOSS, this
% function is useful for more efficient computation time when Jacobian
% calculation is not necessary

function [sum_PHYLOSS,PHYLOSS] = computePHYLOSS(worker,calcNet,calcLib)
    y = calcLib.y(calcNet);         %forward propagation to obtain output
    m_s = size(y{1,1},2);           %number of training series
    ts = size(y,2);                 %number of time steps
    X_inp = worker.param.exInputs;  %exogeneous inputs

    % Get normalization vector for targets
    [errorNorm,Xinputs]  = calcLib.geterrorNorm;    
    
    % Constants for balance equations
    R = 8.314;                      %ideal gas constant J/mol.K
    porosity = X_inp(5,:);          %porosity -
    H_reaction = X_inp(13,:);       %reaction enthalpy J/mol
    M_CaO = 56.0774e-3;             %molar density kg/mol
    rho_CaO  = X_inp(1,:);          %mass density kg/m³
    rho_CaOH2 = X_inp(2,:);         %mass density kg/m³
    cp_CaO = X_inp(3,:);            %CaO specific heat capacity J/kg.K
    cp_CaOH2 = X_inp(4,:);          %Ca(OH)2 specific heat capacity J/kg.K
    moldens_CaO = rho_CaO/M_CaO;    %molar density mol/m^3
    V_sto = 0.08;                   %storage volume m^3
    Z_MB = worker.param.Z_MB;       %normalization quotient for mole balance
    Z_EB = worker.param.Z_EB;       %normalization quotient for energy balance
                
    % Constant values over time
    Q_in = -worker.param.Q_in;      %inflowing heat J
    n_H2O_in = X_inp(12,:)*5;       %inflowing moles*dt(5s) mol

    % Initialize cells for Physical errors
    PHYLOSS{1} = zeros(1,(ts)*m_s);
    PHYLOSS{2} = zeros(1,(ts)*m_s);    
    PHYLOSS{3} = zeros(1,(ts)*m_s);    
    PHYLOSS{4} = zeros(1,(ts)*m_s);
    PHYLOSS{5} = zeros(1,(ts)*m_s);
    PHYLOSS{6} = zeros(1,(ts)*m_s);
    PHYLOSS{7} = zeros(1,(ts)*m_s);
    PHYLOSS{8} = zeros(1,(ts)*m_s);
    PHYLOSS{9} = zeros(1,(ts)*m_s);
    
    % Initial values
    x_CaO = worker.param.startValues(3,:);          % average CaO fraction of storage
    x_CaOH2 = 1 - porosity - x_CaO;                 % average Ca(OH)2 fraction of storage
    x_H2O = worker.param.startValues(4,:);          % average water fraction of storage
    T_sto = worker.param.startValues(2,:);          % average temperature of storage
    p_sto = worker.param.startValues(1,:);          % average pressure of storage
    n_H2O_out = worker.param.startValues(5,:)*5;    % moles of water at ouflow
    Q_out = worker.param.startValues(6,:);          % heat outflow
    moldens_H2O = p_sto./((T_sto)*R);               % H2O molar density
    moldens_gas = moldens_H2O;
    
 	% Calculate the physical loss through time
 
    for t = 1:ts
        % Store values from previous time step
        x_CaO_prev = x_CaO;                 % average CaO fraction of storage
        x_CaOH2_prev = x_CaOH2;             % average Ca(OH)2 fraction of storage
        x_H2O_prev = x_H2O;                 % average water fraction of storage
        T_sto_prev= T_sto;                  % average temp of storage
        p_sto_prev = p_sto;                 % average pressure of storage
        n_H2O_out_prev = n_H2O_out;         % moles of water at ouflow
        Q_out_prev = Q_out;                 % heat outflow
        moldens_H2O_prev = moldens_H2O;     % H2O molar density
        moldens_gas_prev = moldens_gas;
        
        % Update values for current time step
        x_CaO = y{1,t}(3,:);                            % average CaO fraction of storage
        x_CaOH2 = 1 - porosity - x_CaO;                 % average Ca(OH)2 fraction of storage
        x_H2O = y{1,t}(4,:);                            % average water fraction of storage
        T_sto = y{1,t}(2,:);                            % average temp of storage
        p_sto = y{1,t}(1,:);                            % average pressure of storage
        n_H2O_out = worker.param.n_out{1,t}(1,:)*5;     % moles of water at ouflow
        Q_out = worker.param.Q_out{1,t}(1,:);           % heat outflow
        moldens_H2O = p_sto./((T_sto)*R);               % H2O molar density
        moldens_gas = moldens_H2O;

        % Pre-calculation of energy balance error terms
        Q_source_t = V_sto*(x_CaO-x_CaO_prev).*moldens_CaO.*(H_reaction-(porosity./(1-porosity).*p_sto./moldens_gas));   %source term
        dQ_in_out_t = (Q_in-Q_out);      %divergence in/outlet
        dQ_internal_solid_t = V_sto*((cp_CaO .* rho_CaO .* x_CaO + cp_CaOH2 .* rho_CaOH2 .* x_CaOH2).*T_sto-(cp_CaO .* rho_CaO .* x_CaO_prev + cp_CaOH2 .* rho_CaOH2 .* x_CaOH2_prev).*T_sto_prev);        %internal energy of solid

        % Calculate and store each physical error in its respective cell
        PHYLOSS{1}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = (1/Z_MB).* (-n_H2O_out+n_H2O_in +(x_CaO-x_CaO_prev).*moldens_CaO*V_sto+(x_H2O-x_H2O_prev).*porosity*V_sto.*moldens_H2O);    %error to water mole balance
        PHYLOSS{2}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = (1/Z_EB).*(dQ_in_out_t - dQ_internal_solid_t - Q_source_t);   %error to energy balance
        PHYLOSS{3}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(-x_CaO)*errorNorm(3);                %xCaO lower than 0
        PHYLOSS{4}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(-x_H2O)*errorNorm(4);                %xH2O lower than 0
        PHYLOSS{5}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(x_CaO+X_inp(5,:)-1)*errorNorm(3);    %xCaO + porosity > 1
        PHYLOSS{6}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(p_sto_prev-p_sto)*errorNorm(1);      %pressure decrease over time
        PHYLOSS{7}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(T_sto_prev-T_sto)*errorNorm(2);      %temperature decrease over time
        PHYLOSS{8}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(x_CaO_prev-x_CaO)*errorNorm(3);      %xCaO decrease over time
        PHYLOSS{9}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(T_sto-X_inp(10,:))*errorNorm(2);     %Tsto > Tcharge    end
    end
    
    % Normalize all error terms by the number of data points and the output
    for counter = 1:9
        sum_PHYLOSS(counter) = PHYLOSS{counter}*PHYLOSS{counter}'/(ts*m_s);
    end
end