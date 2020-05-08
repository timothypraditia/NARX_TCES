%% Function to calculate Jacobian of the physical error for hyperparameters and network parameters updating

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
% jjPHY: a sum of the approximated Hessian matrix of all physical errors
% (already normalized with lambda)
% jjPHY_i: an array containing an approximation of the Hessian matrix of
% each physical error (not yet normalized with lambda)
% jePHY: a sum of the product of Jacobian and physical error (already
% normalized with lambda)
% sum_PHYLOSS: an array containing the mean squared physical error

function [jjPHY,jjPHY_i,jePHY,sum_PHYLOSS] = computeJPHY(worker,calcNet,calcLib)

    % Get the Jacobian
    [J_perf,e0] = calcLib.simpleJ(calcNet);
    y = calcLib.y(calcNet);        	%forward propagation to obtain output
    m_s = size(y{1,1},2);           %number of training series
    ts = size(y,2);                 %number of time steps
    n_y = size(y{1,1},1);           %number of outputs
    mat_size = m_s * n_y;           %elements of a timestep cell

    % Store a 'delayed' Jacobian matrix 
    J_perf_prev = [zeros(length(worker.WB),mat_size) J_perf];
    J_perf_prev = J_perf_prev(:,1:mat_size*ts);
    X_inp = worker.param.exInputs;        %exogeneous inputs
   
    % Constants for balance equations
    R = 8.314;                      %ideal gas constant J/mol.K
    porosity = X_inp(5,:);          %porosity -
    H_reaction = X_inp(13,:);       %reaction enthalpy J/mol
    M_CaO = 56.0774e-3;             %molar density kg/mol 
    rho_CaO  = X_inp(1,:);          %mass density kg/m^3
    rho_CaOH2 = X_inp(2,:);         %mass density kg/m^3
    cp_CaO = X_inp(3,:);            %mass based heat capacity J/kg.K
    cp_CaOH2 = X_inp(4,:);          %mass based heat capacity J/kg.K
    moldens_CaO = rho_CaO/M_CaO;    %molar density mol/m^3
    V_sto = 0.08;                   %storage volume
    Z_MB = worker.param.Z_MB;       %normalization quotient for mole balance
    Z_EB = worker.param.Z_EB;       %normalization quotient for energy balance
        
    % Constant values over time
    Q_in = -worker.param.Q_in;         %inflowing heat, read from data
    n_H2O_in = X_inp(12,:)*5;  %inflowing moles*dt(5s), exogeneous inputs

    % Get normalization vector for targets
    [errorNorm,Xinputs]  = calcLib.geterrorNorm;

    % Initialize cells for Physical errors and their Jacobians
    PHYLOSS{1} = zeros(1,(ts)*m_s);
    PHYLOSS{2} = zeros(1,(ts)*m_s);    
    PHYLOSS{3} = zeros(1,(ts)*m_s);    
    PHYLOSS{4} = zeros(1,(ts)*m_s);
    PHYLOSS{5} = zeros(1,(ts)*m_s);
    PHYLOSS{6} = zeros(1,(ts)*m_s);
    PHYLOSS{7} = zeros(1,(ts)*m_s);
    PHYLOSS{8} = zeros(1,(ts)*m_s);
    PHYLOSS{9} = zeros(1,(ts)*m_s);
    jPHY{1} = zeros(size(J_perf,1),(ts)*m_s);
    jPHY{2} = zeros(size(J_perf,1),(ts)*m_s);
    jPHY{3} = zeros(size(J_perf,1),(ts)*m_s);
    jPHY{4} = zeros(size(J_perf,1),(ts)*m_s);
    jPHY{5} = zeros(size(J_perf,1),(ts)*m_s);
    jPHY{6} = zeros(size(J_perf,1),(ts)*m_s);
    jPHY{7} = zeros(size(J_perf,1),(ts)*m_s);
    jPHY{8} = zeros(size(J_perf,1),(ts)*m_s);
    jPHY{9} = zeros(size(J_perf,1),(ts)*m_s);
    
    % Iinitial values
    x_CaO = worker.param.startValues(3,:);          % average CaO fraction of storage
    x_CaOH2 = 1 - porosity - x_CaO;                 % average Ca(OH)2 fraction of storage
    x_H2O = worker.param.startValues(4,:);          % average water fraction of storage
    T_sto = worker.param.startValues(2,:);          % average temp of storage
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
        n_H2O_out = worker.param.n_out{1,t}(1,:)*5;     % moles of water at outflow
        Q_out = worker.param.Q_out{1,t}(1,:);           % heat outflow
        moldens_H2O = p_sto./((T_sto)*R);               % H2O molar density
        moldens_gas = moldens_H2O;
        
        %derivatives of each target with respect to the weights (prev
        %for previous time step)
        dp = -J_perf (:,((t-1)*mat_size+0*m_s+1:(t-1)*mat_size+0*m_s+m_s))/errorNorm(1);
        dp_prev = -J_perf_prev (:,((t-1)*mat_size+0*m_s+1:(t-1)*mat_size+0*m_s+m_s))/errorNorm(1);
        dT = -J_perf (:,((t-1)*mat_size+1*m_s+1:(t-1)*mat_size+1*m_s+m_s))/errorNorm(2);
        dT_prev = -J_perf_prev (:,((t-1)*mat_size+1*m_s+1:(t-1)*mat_size+1*m_s+m_s))/errorNorm(2);
        dCaO = -J_perf (:,((t-1)*mat_size+2*m_s+1:(t-1)*mat_size+2*m_s+m_s))/errorNorm(3);
        dCaO_prev = -J_perf_prev (:,((t-1)*mat_size+2*m_s+1:(t-1)*mat_size+2*m_s+m_s))/errorNorm(3);
        dH2O = -J_perf (:,((t-1)*mat_size+3*m_s+1:(t-1)*mat_size+3*m_s+m_s))/errorNorm(4);
        dH2O_prev = -J_perf_prev (:,((t-1)*mat_size+3*m_s+1:(t-1)*mat_size+3*m_s+m_s))/errorNorm(4);
        dn_out = 0;
        dQ_out = 0;

        % Pre-calculation of energy balance error terms
        Q_source_t = -V_sto*(x_CaO-x_CaO_prev).*moldens_CaO.*(H_reaction-(porosity./(1-porosity).*p_sto./moldens_gas));   %source term
        dQ_in_out_t = (-Q_in+Q_out);      %divergence in/outlet
        dQ_internal_solid_t = V_sto*((cp_CaO .* rho_CaO .* x_CaO + cp_CaOH2 .* rho_CaOH2 .* x_CaOH2).*T_sto-(cp_CaO .* rho_CaO .* x_CaO_prev + cp_CaOH2 .* rho_CaOH2 .* x_CaOH2_prev).*T_sto_prev);        %internal energy of solid

        % Calculate and store each physical error in its respective cell
        PHYLOSS{1}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = (1/Z_MB).* (n_H2O_out-n_H2O_in -(x_CaO-x_CaO_prev).*moldens_CaO*V_sto+(x_H2O-x_H2O_prev).*porosity*V_sto.*moldens_H2O);    %error to water molar balance
        PHYLOSS{2}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = (1/Z_EB).*(dQ_in_out_t + dQ_internal_solid_t - Q_source_t);   %error to energy balance
        PHYLOSS{3}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(-x_CaO)*errorNorm(3);             %xCaO lower than 0
        PHYLOSS{4}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(-x_H2O)*errorNorm(4);             %xH2O lower than 0
        PHYLOSS{5}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(x_CaO+X_inp(5,:)-1)*errorNorm(3); %xCaO + porosity > 1
        PHYLOSS{6}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(p_sto_prev-p_sto)*errorNorm(1);  %pressure decrease over time
        PHYLOSS{7}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(T_sto_prev-T_sto)*errorNorm(2);   %temperature decrease over time
        PHYLOSS{8}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(x_CaO_prev-x_CaO)*errorNorm(3);   %xCaO decrease over time
        PHYLOSS{9}(1,((t-1)*m_s+1:(t-1)*m_s+m_s)) = poslin(T_sto-X_inp(10,:))*errorNorm(2);  %Tsto > Tcharge

        % Calculate and store the Jacobian in its respective cell
        jPHY{1}(:,((t-1)*m_s+1:(t-1)*m_s+m_s)) = (1/Z_MB)*(dn_out - bsxfun(@times,(dCaO-dCaO_prev), moldens_CaO*V_sto) + bsxfun(@times,(dH2O-dH2O_prev), porosity*V_sto.*moldens_H2O)-bsxfun(@times,dT,(x_H2O-x_H2O_prev).*porosity*V_sto/R.*p_sto./((T_sto).^2))+bsxfun(@times,dp,(x_H2O-x_H2O_prev).*porosity*(V_sto/R)./T_sto));
        jPHY{2}(:,((t-1)*m_s+1:(t-1)*m_s+m_s)) = (1/Z_EB)*(dQ_out + bsxfun(@times,dT,V_sto*((cp_CaO .* rho_CaO .* x_CaO + cp_CaOH2 .* rho_CaOH2 .* x_CaOH2)))-bsxfun(@times,dT_prev,V_sto*((cp_CaO .* rho_CaO .* x_CaO_prev + cp_CaOH2 .* rho_CaOH2 .* x_CaOH2_prev)))+bsxfun(@times,dCaO,(cp_CaO .* rho_CaO-cp_CaOH2 .* rho_CaOH2).*T_sto)-bsxfun(@times,dCaO_prev,(cp_CaO .* rho_CaO-cp_CaOH2 .* rho_CaOH2).*T_sto_prev)+bsxfun(@times,(dCaO -dCaO_prev),moldens_CaO*V_sto.*(H_reaction-((porosity./(1-porosity)).*p_sto./moldens_gas))) -bsxfun(@times,dT,V_sto*(x_CaO-x_CaO_prev).*moldens_CaO.*(porosity./(1-porosity))*R)); 
        jPHY{3}(:,((t-1)*m_s+1:(t-1)*m_s+m_s))= bsxfun(@times,(-dCaO)*errorNorm(3),hardlim(-x_CaO-1E-10));
        jPHY{4}(:,((t-1)*m_s+1:(t-1)*m_s+m_s))= bsxfun(@times,(-dH2O)*errorNorm(4),hardlim(-x_H2O-1E-10));
        jPHY{5}(:,((t-1)*m_s+1:(t-1)*m_s+m_s))= bsxfun(@times,(dCaO)*errorNorm(3),hardlim(x_CaO+X_inp(5,:)-1-1E-10));
        jPHY{6}(:,((t-1)*m_s+1:(t-1)*m_s+m_s))= bsxfun(@times,(dp_prev-dp)*errorNorm(1),hardlim(p_sto_prev-p_sto-1E-10));
        jPHY{7}(:,((t-1)*m_s+1:(t-1)*m_s+m_s))= bsxfun(@times,(dT_prev-dT)*errorNorm(2),hardlim(T_sto_prev-T_sto-1E-10));
        jPHY{8}(:,((t-1)*m_s+1:(t-1)*m_s+m_s))= bsxfun(@times,(dCaO_prev-dCaO)*errorNorm(3),hardlim(x_CaO_prev-x_CaO-1E-10));
        jPHY{9}(:,((t-1)*m_s+1:(t-1)*m_s+m_s))= bsxfun(@times,(dT)*errorNorm(2),hardlim(T_sto-X_inp(10,:)-1E-10));
    end
    
    % Normalization of Jacobian
    jPHY{1} = jPHY{1}/(ts*m_s);
    jPHY{2} = jPHY{2}/(ts*m_s);    
    jPHY{3} = jPHY{3}/(ts*m_s);  
    jPHY{4} = jPHY{4}/(ts*m_s);  
    jPHY{5} = jPHY{5}/(ts*m_s);  
    jPHY{6} = jPHY{6}/(ts*m_s);  
    jPHY{7} = jPHY{7}/(ts*m_s);  
    jPHY{8} = jPHY{8}/(ts*m_s);      
    jPHY{9} = jPHY{9}/(ts*m_s);   
    
    for counter = 1:9
        sum_PHYLOSS(counter) =PHYLOSS{counter}*PHYLOSS{counter}'/(ts*m_s); 
        jjPHY_i{counter} = jPHY{counter}*jPHY{counter}';
    end
    
    jjPHY = worker.lambdaPHY(1)*jPHY{1}*jPHY{1}'+ worker.lambdaPHY(2)*jPHY{2}*jPHY{2}'+...
        worker.lambdaPHY(3)*jPHY{3}*jPHY{3}'+ worker.lambdaPHY(4)*jPHY{4}*jPHY{4}'+...
        worker.lambdaPHY(5)*jPHY{5}*jPHY{5}'+ worker.lambdaPHY(6)*jPHY{6}*jPHY{6}'+...
        worker.lambdaPHY(7)*jPHY{7}*jPHY{7}'+ worker.lambdaPHY(8)*jPHY{8}*jPHY{8}'+...
        worker.lambdaPHY(9)*jPHY{9}*jPHY{9}';
    jePHY = worker.lambdaPHY(1)*jPHY{1}*PHYLOSS{1}'+ worker.lambdaPHY(2)*jPHY{2}*PHYLOSS{2}'+...
        worker.lambdaPHY(3)*jPHY{3}*PHYLOSS{3}'+ worker.lambdaPHY(4)*jPHY{4}*PHYLOSS{4}'+...
        worker.lambdaPHY(5)*jPHY{5}*PHYLOSS{5}'+ worker.lambdaPHY(6)*jPHY{6}*PHYLOSS{6}'+...
        worker.lambdaPHY(7)*jPHY{7}*PHYLOSS{7}'+ worker.lambdaPHY(8)*jPHY{8}*PHYLOSS{8}'+...
        worker.lambdaPHY(9)*jPHY{9}*PHYLOSS{9}';

    % Print value of physical error each 10th epoch
    if (mod(worker.epoch-1,10)==0)
        fprintf(strcat(['PHY LOSS: ',num2str(sum_PHYLOSS),'\n']));
        fprintf(strcat(['epoch: ',num2str(worker.epoch),'\n\n']));
    end

end