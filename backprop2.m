function backprop2(net)

    images = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');

    % disp(images);
    % display_network(images(:,1:100)); % Show the first 100 images
    % disp(labels(1:10));

    eta = 0.5;

    N = size(images);
    N = N(2);

    % numero di strati della rete
    len = length(net.hiddenSize);

    targets = zeros(10, N);

    for i = 1: N
        targets(labels(i, 1)+1, i) = 1;
    end


    for im = 1: 10

        x = images(:, im);
        delta = cell(len-1, 1);
        outputLayers = forwardpropagation(net, x);

        for i = 1: net.hiddenSize(end)
            delta{end}(i) = (outputLayers{end}(i) - targets(i));
        end
    
        % Per ogni layer calcolo il delta
        for layer = len-1 : -1 : 2

            for node = 1 : net.hiddenSize(layer)
                % Calcolo delta
                delta{layer-1}(node) = outputLayers{layer-1}(node) * (1 - outputLayers{layer-1}(node));
                % Estraggo i pesi in uscita dal nodo
                exitWeights = net.weights{layer}(:, node);
                % Estraggo i delta del layer successivo
                deltaNext =  delta{layer};
                delta{layer-1}(node) = delta{layer-1}(node) * sum(exitWeights .* deltaNext');
            end
        end

        % Aggiornamento pesi
        for layer = 1 : length(net.hiddenSize) - 1

            w = net.weights{layer};
            col = size(w);
            col = col(2);

            for j = 1 : net.hiddenSize(layer+1)

                for k = 1 : col
                    x = net.weights{layer}(j,k);
                    y = eta * delta{layer}(j) * outputLayers{layer}(j);
                    z = x - y;
                    
                    disp(net.weights{layer}(j,k));
                    disp(layer);
                    disp(j);
                    disp(k);
                    disp(z);
                    net.weights{layer}(j,k) = z;
                end
            end
        end

    end
end







