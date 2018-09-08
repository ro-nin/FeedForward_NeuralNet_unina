function net = feedforwardnet(hiddenSizes,trainFnc)
    
net.hiddenSizes = hiddenSizes;
net.trainFnc = trainFnc;

for i = 1: length(hiddenSizes) - 1
    net.weights{i} = normrnd(0, max(hiddenSizes)^(-0.5), hiddenSizes(i+1), hiddenSizes(i));
    net.biases{i} = zeros(hiddenSizes(i+1), 1);
end

end