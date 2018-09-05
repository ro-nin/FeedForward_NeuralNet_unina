function backprop(net)

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


    for i = 1: 10

        x = images(:, i);
        delta = cell(len, 1);
        y = forwardpropagation(net, x);

        %Backprop
        for k = 1 : net.hiddenSize(len)
            delta{len}(k) = (y{len}(k) - targets(k));
        end

        for i = len - 1 : -1 : 2

            for j = 1 : net.hiddenSize(i)            
                delta{i}(j) = y{i}(j) .* (1 - y{i}(j));            
                matrix = net.weights{i};
                column = matrix(:, j);
                delta{i}(j) = sum(column' .* delta{i+1});
            end
        end

        for i = 1 : length(net.hiddenSize) - 1

            w = net.weights{i};
            col = size(w);
            col = col(2);

            for j = 1 : net.hiddenSize(i+1)

                for k = 1 : col
                    net.weights{i}(j,k) = net.weights{i}(j,k) - (eta * delta{i+1}(j) * y{i+1}(j));
                end
            end

        end

    end
end







