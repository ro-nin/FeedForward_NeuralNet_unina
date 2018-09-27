
test_im = loadMNISTImages('t10k-images.idx3-ubyte');
test_lb = loadMNISTLabels('t10k-labels.idx1-ubyte');
test_lb = test_lb';

test_lb(test_lb==0) = 10;
test_lb = dummyvar(test_lb);


error = 0;
guessed = 0;

for j = 1: 10000  
    test = propagate(net, test_im(:, j)');
    [val, idx] = max(test{end});
    if(idx == find(test_lb(j, :)))
        guessed = guessed + 1;
    end
    error = error + 0.5 * (sum(test{end} - test_lb(j,:)')^2);
end

accuracy = (guessed/10000) * 100;
fprintf("0; %.2f%%; %f\n", accuracy, error);



