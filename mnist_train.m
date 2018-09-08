function out = mnist_train(net, epochs)
    
images = loadMNISTImages('train-images.idx3-ubyte');
labels = loadMNISTLabels('train-labels.idx1-ubyte');
labels = labels';

labels(labels==0) = 10;                                
labels = dummyvar(labels);

sizeoftrain = 100;

for i = 1: epochs
    for im = 1: sizeoftrain
        net = train(net, images(:, im)', labels(im, :)', 0.1);
    end
end

guessed = 0;
error = 0;

for i = 1: sizeoftrain    
    test = propagate(net, images(:, i)');    
    [val, idx] = max(test{end});
    %fprintf("val: %.2f - idx: %d\n", val, idx);    
    if(idx == find(labels(i, :)))
        guessed = guessed + 1;
    end   
    error = error + sum((test{end} - labels(i,:)').^2);    
end

error = error * 0.5;

fprintf("guessesed: %d/%d rate: %.2f%%\n", guessed, sizeoftrain,(guessed/sizeoftrain) * 100);
fprintf("total error: %f\n", error);

%display_network(images(:,1:100));
out = net;

end