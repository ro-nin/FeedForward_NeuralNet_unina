function out = simple_back(net, input, targets)

output = propagate(net, input);
outputLayer = output{end};
delta2 = zeros(net.hiddenSizes(end), 1);

for i = 1: net.hiddenSizes(end)
    delta2(i) = (outputLayer(i) - targets(i));
end

delta1 = zeros(net.hiddenSizes(2), 1);

hiddenLayer = output{2};

for j = 1: net.hiddenSizes(2)
    
    delta1(j) = hiddenLayer(j)*(1 - hiddenLayer);
    
    temp = 0;
    
    for k = 1: net.hiddenSizes(end)
        
        temp = temp + net.weights{2}(k, j) * delta2(k);
        
    end
    
    delta1(j) = temp * delta1(j);

end

for k = 1: net.hiddenSizes(3)
   for j = 1: net.hiddenSizes(2)
        net.weights{2}(k, j) =  net.weights{2}(k, j) - 0.1 * delta2(k) * hiddenLayer(j);
   end
end

inputLayer = output{1};

for j = 1: net.hiddenSizes(2)
    for i = 1: net.hiddenSizes(1)
        net.weights{1}(j, k) =  net.weights{1}(j, k) - 0.1 * delta1(j)* inputLayer(i);
    end
end
    out = net;
end












