function out = train(net, size, input, targets, eta, errorFnc, isOnline)

if ( isOnline == 1)
     for i = 1: size
        [dWeights, dBiases] = backPropagation(net, input(i, :), targets(i, :), errorFnc);
        [net, deltaW, deltaB] = gradientDescent(net, dWeights, dBiases, eta);
     end    
else
    [dWeights, dBiases] = backPropagation(net, input, targets, errorFnc);
    [net, deltaW, deltaB] = gradientDescent(net, dWeights, dBiases, eta);
end
    out = net;
end