function out = propagate(net, input)
    
out = cell(length(net.hiddenSizes), 1);
out{1} = input';

for i = 1: length(net.hiddenSizes) - 1
    out{i+1} = net.trainFnc{i}((net.weights{i} * out{i}) + net.biases{i});
end

end