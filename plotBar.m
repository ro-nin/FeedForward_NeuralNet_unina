function plotBar(fnc1, fnc2, netNodes, netEtas, deviationsNodes)
%Plot K-fold error measure for given functions and hyperparams
%   the plot describes the cross validation results grouped by hidden nodes
%   number used, with each bar represeting the error standard deviation on
%   given learning rate
%
%fnc1: activation function for hidden nodes layer
%
%fnc2: activation function for output nodes layer
%
%netNodes: cell array with number of hidden nodes used for the CV,
%          ex:({250},{500},{800})
%
%netEtas: cell array containing the learning rates used in the CV
%
%deviationNodes: counter for how many hidden nodes number are there
graphName = sprintf('K-Fold: %s - %s', fnc1, fnc2);

figure('Name', graphName);

c = categorical(netNodes);

y = zeros(length(netNodes), length(netEtas));

for devNodesIndex = 1 : length(deviationsNodes)
    tmp = cell2mat(deviationsNodes{devNodesIndex})';
    y(devNodesIndex, :) = tmp;
end

b = bar(c,y);

title(graphName);

l = cell(1, length(netEtas));
for i = 1: length(netEtas)
    l{i} = sprintf('%.4f',netEtas(i));
end

legend(b, l);

xlabel('Hidden Nodes');
ylabel('Mean Accuracy');



a = (1:size(y,1)).';
x = [a-0.20 a a+0.20 a+0.40];
for k=1:size(y,1)
    for m = 1:size(y,2)
        text(x(k,m),y(k,m),num2str(y(k,m),'%0.2f'),...
            'HorizontalAlignment','center',...
            'VerticalAlignment','bottom')
    end
end

drawnow;

end