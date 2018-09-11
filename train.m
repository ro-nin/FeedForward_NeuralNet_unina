function out = train(net, x, t, eta)
%funzione utilizzata per addestrare la rete fornita su un singolo esempio del training set.
    %net: la rete da addestrare
    %x: input
    %t: vettore target nella forma (0,0,0,0,1,0,0,0,0,0)
    %eta: learning rate
    %out: la rete net, con i pesi aggiornati

%forward propagation
outputs = propagate(net, x);
%cell array contenente per ogni strato, i delta di ogni nodo
delta = cell(length(net.hiddenSizes) - 1, 1);
%calcolo dell'errore tramite funzione di costo specificata
delta{end}=net.costFnc(outputs{end},t);

%calcolo delle derivate per il gradiente
for i = length(net.hiddenSizes) - 1: - 1: 2
    %prelievo dell'handle per la derivata adeguata
    deriv=net.trainDerFnc{i-1};
    delta{i-1} = (((net.weights{i}' * delta{i})') .* deriv(outputs{i}'))';
end

%aggiornamento dei pesi 
for l = 1 : length(net.hiddenSizes) - 1
    for i = 1: net.hiddenSizes(l+1)
        for j = 1: net.hiddenSizes(l)
            net.weights{l}(i, j) = net.weights{l}(i, j) - eta * delta{l}(i) * outputs{l}(j);
        end
        net.biases{l}(i) = net.biases{l}(i) - eta * delta{l}(i);
    end
end

out = net;

end
