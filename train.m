function out = train(net, input, targets, eta, TsSize, batchSize)

 for i = 1: batchSize: TsSize
    [dWeights, dBiases] = ...
        backPropagation(net, input(i: i+batchSize-1, :), targets(i:i+batchSize-1, :));
    net = gradientDescent(net, dWeights, dBiases, eta);
 end
 
    out = net;
    
end