function net = feedforwardnet(hiddenSizes,trainFnc,costFnc)
%Funzione che crea una rete neurale artificiale feed forward multistrato
%full connected
    %hiddenSizes: vettore che contiene in ogni posizione, il numero di
                %input in posizione 1, e per il numero di nodi interni per le posizioni
                %intermedie e il numero di nodi output in ultima posizione, es: [784,200,10]
    %trainFnc: array cell contenente gli handle delle funzioni di attivazioni
                %per l'i-esimo strato, es: {@sigmoid,@identity}.
    %oostFnc: handle contenente la funzione di errore per lo strato di
                %output, es: @quadratiCost.
                
                
net.hiddenSizes = hiddenSizes;
net.trainFnc = trainFnc;
net.trainDerFnc = cell(length(trainFnc), 1);

%controlla se l'utente ha inserito la funzione di costo
if (~exist('costFnc','var'))
    net.costFnc=@quadraticCost;
    disp("No cost function specified, setting to quadratic");
else
    net.costFnc = costFnc;
end

%associa le derivate delle funzioni alle attivazione
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

%inizializzazione dei pesi in modo casuale con distribuzione normale
for i = 1: length(hiddenSizes) - 1
    net.weights{i} = normrnd(0, max(hiddenSizes)^(-0.5), hiddenSizes(i+1), hiddenSizes(i));
    %inizializzazione a 0 dei bias
    net.biases{i} = zeros(hiddenSizes(i+1), 1);
end

end