function backprop(net, imagename, labelname)

    for k = 1 : net.hiddenSize(len)
        delta{len}(k) = (outputLayer{len}(k) - targets(k));
    end

    for i = len - 1 : -1 : 2

        for j = 1 : net.hiddenSize(i)

            delta{i}(j) = outputLayer{i} * (1 - outputLayer{i});

            matrix = weight = net.weights{i+1};
            column = matrix(:, j);

            delta{i}(j) = sum(column .* delta{i+1}(j));
        end

    end

    for i = 1 : length(net.hiddenSize)

        for j = 1 : net.hiddenSize(i)
            net.weights{i}(j) = net.weights{i}(j) - eta * delta{i}(j) * outputLayer{i}(j);
        end

    end
end