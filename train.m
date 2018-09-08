function out = train(net, x, t, eta)

outputs = propagate(net, x);
delta = cell(length(net.hiddenSizes) - 1, 1);
delta{end} =  outputs{end} - t;

for i = length(net.hiddenSizes) - 1: - 1: 2
    delta{i-1} = ((net.weights{i}' * delta{i})') .* outputs{i}' .*  (1 - outputs{i}');
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