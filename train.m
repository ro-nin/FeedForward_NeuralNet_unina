function out = train(net, size, input, targets, eta, isOnline)

if ( isOnline == 1)
     for i = 1: size
        [dWeights, dBiases] = backPropagation(net, input(i, :), targets(i, :));
        net = gradientDescent(net, dWeights, dBiases, eta);
     end    
else
    [dWeights, dBiases] = backPropagation(net, input, targets);
    net = gradientDescent(net, dWeights, dBiases, eta);
end
    out = net;
end