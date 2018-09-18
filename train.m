function out_net = train(net, input, targets, eta)

% input: matrice delle immagini (num_immagini x num_features)
% targets: matrice dei targets (num_uscite x num_immagini)

delta = cell(length(net.hiddenSizes) - 1, 1);
accDeltaW = cell(length(net.hiddenSizes) - 1, 1);
accDeltaB = cell(length(net.hiddenSizes) - 1, 1);

for i = 1: length(net.hiddenSizes) - 1
    accDeltaW{i} = zeros(net.hiddenSizes(i+1), net.hiddenSizes(i));
    accDeltaB{i} = zeros(net.hiddenSizes(i+1), 1);
end

for im = 1: 1 % size(input, 1)
    % eseguo la feed forward propagation
    outputs = propagate(net, input(im, :));
    
    % estraggo la derivata dell'output e la calcolo
    derivate = net.trainDerFnc{end}(outputs{end});
    delta{end} = net.costFnc(outputs{end}, targets(:, im)) .* derivate;
    
    % delta: array cell di vettori colonna dei delt di ciascun nodo
    
    %calcolo delle derivate per il gradiente
    for i = length(net.hiddenSizes) - 1: - 1: 2
        %prelievo dell'handle per la derivata e la eseguo
        derivate = net.trainDerFnc{i-1}(outputs{i}');
        delta{i-1} = (((net.weights{i}' * delta{i})') .* derivate)';
    end
    
    for l = 1 : length(net.hiddenSizes) - 1
        for i = 1: net.hiddenSizes(l+1)
            for j = 1: net.hiddenSizes(l)
                accDeltaW{l}(i, j) = accDeltaW{l}(i, j) - eta * delta{l}(i) * outputs{l}(j);
            end
            accDeltaB{l}(i) = accDeltaB{l}(i) - eta * delta{l}(i);
        end
    end   
end

% aggiornamento dei pesi
for i = 1: length(net.hiddenSizes) - 1
    net.weights{i} = net.weights{i} + accDeltaW{i};
    net.biases{i} = net.biases{i} + accDeltaB{i};
end

out_net = net;

end