function out_net = train(net, input, targets, eta)

% input: matrice delle immagini (num_esempi x num_features)
% targets: matrice dei targets (num_uscite x num_esempi)

grad = cell(length(net.hiddenSizes) - 1, 1);
deltaWeights = cell(length(net.hiddenSizes) - 1, 1);
deltaBiases = cell(length(net.hiddenSizes) - 1, 1);

for i = 1: length(net.hiddenSizes) - 1
    deltaWeights{i} = zeros(net.hiddenSizes(i+1), net.hiddenSizes(i));
    deltaBiases{i} = zeros(net.hiddenSizes(i+1), 1);
end


%for im = 1: size(input, 1)
    tic
    % eseguo la feed forward propagation
    %outputs = propagate(net, input(im, :));
    outputs = propagate(net, input);
    
    % estraggo la derivata dell'output e la calcolo
    derivate = net.trainDerFnc{end}(outputs{end});
    %grad{end} = net.costFnc(outputs{end}, targets(:, im)) .* derivate;
    grad{end} = net.costFnc(outputs{end}, targets) .* derivate;
    
    % calcolo delle derivate per il gradiente
    for i = length(net.hiddenSizes) - 1: - 1: 2
        %prelievo dell'handle per la derivata e la eseguo
        derivate = net.trainDerFnc{i-1}(outputs{i}');
        grad{i-1} = (((net.weights{i}' * grad{i})') .* derivate)';
    end
    
    for l = 1 : length(net.hiddenSizes) - 1
        for i = 1: net.hiddenSizes(l+1)
            for j = 1: net.hiddenSizes(l)
                deltaWeights{l}(i, j) = deltaWeights{l}(i, j) - grad{l}(i) * outputs{l}(j);
            end
            deltaBiases{l}(i) = deltaBiases{l}(i) - grad{l}(i);
        end
    end
    toc
%end

% aggiornamento dei pesi
for i = 1: length(net.hiddenSizes) - 1
    net.weights{i} = net.weights{i} + eta .* deltaWeights{i};
    net.biases{i} = net.biases{i} + eta .* deltaBiases{i};
end

out_net = net;

end