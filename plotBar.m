function plotBar(fnc1, fnc2, netNodes, netEtas, deviationsNodes)

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
    l{i} = sprintf('%.3f',netEtas(i));
end

legend(b, l);

xlabel('Hidden Nodes');
ylabel('Error Standard Deviation');
drawnow;

end