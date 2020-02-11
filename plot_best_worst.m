function plot_best_worst(netc,X_test,Y_test,feedback_delay)

    [Xc,Xic,Aic,~] = preparets(netc,X_test,{},Y_test);
    y_NARX = netc(Xc,Xic,Aic);

    num_dataset = size(y_NARX{1},2);
    timestep = size(y_NARX,2);

    y_pred = zeros(4,timestep,num_dataset);
    y_num = zeros(4,timestep,num_dataset);
    MSE_best = 1E3;
    MSE_worst = 0;

    for t = 1:timestep
        y_pred(:,t,:) = y_NARX{t};
        y_num(:,t,:) = Y_test{t+feedback_delay};
    end

    outputInd = find(netc.outputConnect);
    numOutputs = length(outputInd);
    eInd = 1:numOutputs;

    ii = 1;
    i = outputInd(ii);
    eind = eInd(ii);
    range = netc.outputs{i}.range;
    rMin = range(:,1);
    rMax = range(:,2);
    ratio = 2 ./ (rMax - rMin);
    ratio(~isfinite(ratio)) = 1;
    ratio = ratio(:,ones(1,timestep));

    for t = 1:num_dataset
        diff = gsubtract(y_pred(:,:,t),y_num(:,:,t));
        e = diff.*ratio;
        MSE(t) = mean(e(:).^2);
        if MSE_worst < MSE(t)
            MSE_worst = MSE(t);
            worst_index = t;
        end
        if MSE_best > MSE(t)
            MSE_best = MSE(t);
            best_index = t;
        end
    end

    %% Plotting Worst

    figure
    set(gcf,'units','centimeters','position',[5,5,12,9])
    set(groot, 'DefaultTextInterpreter', 'LaTeX');
    set(groot, 'DefaultAxesTickLabelInterpreter', 'LaTeX');
    set(groot, 'DefaultAxesFontName', 'LaTeX');
    set(groot, 'DefaultLegendInterpreter', 'LaTeX');
    
    posnew = [0.125,0.6,0.325,0.333; 0.6,0.6,0.325,0.333; 0.125,0.175,0.325,0.333; 0.6,0.175,0.325,0.333];

    for i = 1:4
        subplot(2,2,i)
        set(subplot(2,2,i),'position',posnew(i,:))
        set(gca,'FontSize',10)
        box on;

        hold on;
        plot(y_num(i,:,best_index),'b-','LineWidth',1,'Marker','o','MarkerSize',6,...
            'MarkerIndices',1:50:995,'DisplayName','Reference');
        plot(y_pred(i,:,best_index),'r:','LineWidth',1,'MarkerSize',6,'DisplayName','ANN');
        set(gca,'xlim',([0 1000]))
        set(gca,'xtick',([min(xlim):200:max(xlim)]))

        xlabel('time step')
        ylabel_strings={'pressure/ Pa','temperature/ K','$\upsilon_{CaO}$','$x_{g,H2O}$'};
        ylabel(ylabel_strings{1,i});

    end

    set(gcf, 'Position',  [0, 0, 20, 15])

    lgd = legend;
    lgd.NumColumns = 2;
    lgd.FontSize = 12

    set(lgd, 'Position', [0.25,0.02,0.5,0.05]);

    savefig('best.fig');
    
    figure
    set(gcf,'units','centimeters','position',[5,5,12,9])
    set(groot, 'DefaultTextInterpreter', 'LaTeX');
    set(groot, 'DefaultAxesTickLabelInterpreter', 'LaTeX');
    set(groot, 'DefaultAxesFontName', 'LaTeX');
    set(groot, 'DefaultLegendInterpreter', 'LaTeX');
    
	for i = 1:4
        subplot(2,2,i)
        set(subplot(2,2,i),'position',posnew(i,:))
        set(gca,'FontSize',10)
        box on;

        hold on;
        plot(y_num(i,:,worst_index),'b-','LineWidth',1,'Marker','o','MarkerSize',6,...
            'MarkerIndices',1:50:995,'DisplayName','Reference');
        plot(y_pred(i,:,worst_index),'r:','LineWidth',1,'MarkerSize',6,'DisplayName','ANN');

        set(gca,'xlim',([0 1000]))
        set(gca,'xtick',([min(xlim):200:max(xlim)]))

        xlabel('time step')
        ylabel_strings={'pressure/ Pa','temperature/ K','$\upsilon_{CaO}$','$x_{g,H2O}$'};
        ylabel(ylabel_strings{1,i});

    end

    set(gcf, 'Position',  [0, 0, 20, 15])

    lgd = legend;
    lgd.NumColumns = 2;
    lgd.FontSize = 12

    set(lgd, 'Position', [0.25,0.02,0.5,0.05]);

    savefig('worst.fig');

end