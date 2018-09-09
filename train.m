function out = train(net, x, t, eta)
    
syms f(h);

%forward propagation
outputs = propagate(net, x);

%inizializzo struttura per i delta
delta = cell(length(net.hiddenSizes) - 1, 1);

%delta dello strato di output
delta{end} =  outputs{end} - t;

%backprop
for i = length(net.hiddenSizes) - 1: - 1: 2
    %calcolo derivata della funzione di attivazione del layer corrente
    
    %Per il calcolo della derivata della funzione di attivazione

    activationFunction=net.trainFnc{i-1};
    %disp(activationFunction);
    f(h)= activationFunction(h);
    %disp(f(h));
    df=diff(f,h);
    %disp(df);
    
    %aggiorno delta del layer corrente
    %delta{i-1} = ((net.weights{i}' * delta{i})') .* outputs{i}' .*  (1 - outputs{i}');
    for nodeInd = 1: net.hiddenSizes(i)
       df2=df(outputs{i-1}(nodeInd));
       deltaPrev=double(df2);
       outWeights=net.weights{i};
       delta{i-1}(nodeInd)=(outWeights(:,nodeInd)' * delta{i}) *  deltaPrev;
    end
end

for l = 1 : length(net.hiddenSizes) - 1
    for i = 1: net.hiddenSizes(l+1)
        for j = 1: net.hiddenSizes(l)
            %printf("L: %d\t I: %d\t J: %d\n", l, i, j);
            net.weights{l}(i, j) = net.weights{l}(i, j) - eta * delta{l}(i) * outputs{l}(j);
        end
        net.biases{l}(i) = net.biases{l}(i) - eta * delta{l}(i);
    end
end

out = net;

end
