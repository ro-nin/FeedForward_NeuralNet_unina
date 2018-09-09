function net = feedforwardnet(hiddenSizes,trainFnc)
    
net.hiddenSizes = hiddenSizes;
net.trainFnc = trainFnc;
net.trainDerFnc = cell(length(trainFnc), 1);

%associa le derivate delle funzioni alle attivazioni
for i = 1: length(trainFnc)
    if isequal(net.trainFnc{i},@sigmoid)
        targetFnc=@sigmoidDerivative;
    elseif isequal(net.trainFnc{i},@identity)
        targetFnc=@identityDerivative;
    elseif isequal(net.trainFnc{i},@tanH)
        targetFnc=@tanHDerivative;
    elseif isequal(net.trainFnc{i},@ReLU)
        targetFnc=@ReLUDerivative;
    else
        disp("errore funzioni di attivazione");
    end
        
    net.trainDerFnc{i}=targetFnc;
end

for i = 1: length(hiddenSizes) - 1
    net.weights{i} = normrnd(0, max(hiddenSizes)^(-0.5), hiddenSizes(i+1), hiddenSizes(i));
    net.biases{i} = zeros(hiddenSizes(i+1), 1);
end

end