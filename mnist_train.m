function out = mnist_train(net, epochs,sizeoftrain,sizeoftest,eta)
    
train_im = loadMNISTImages('train-images.idx3-ubyte');
train_lb = loadMNISTLabels('train-labels.idx1-ubyte');
train_lb = train_lb';

train_lb(train_lb==0) = 10;                                
train_lb = dummyvar(train_lb);

test_im = loadMNISTImages('t10k-images.idx3-ubyte');
test_lb = loadMNISTLabels('t10k-labels.idx1-ubyte');
test_lb = test_lb';

test_lb(test_lb==0) = 10;                                
test_lb = dummyvar(test_lb);

if nargin < 5
    fprintf("not enough params, setting to default\n");
    epochs=1;
    sizeoftrain = 100;
    sizeoftest = 10000;
    eta=0.01;
end
%start counting execution time
tic
for i = 1: epochs
    for im = 1: sizeoftrain
        fprintf("ep:%d, im:%d \n",i,im);
        net = train(net, train_im(:, im)', train_lb(im, :)', eta);
    end
end

guessed = 0;
error = 0;

for i = 1: sizeoftest    
    test = propagate(net, test_im(:, i)');    
    [val, idx] = max(test{end});
    %fprintf("val: %.2f - idx: %d\n", val, idx);    
    if(idx == find(test_lb(i, :)))
        guessed = guessed + 1;
    end   
    error = error + 0.5 * sum((test{end} - test_lb(i,:)').^2);    
end

fprintf("epoch: %d, training samples: %d\n", epochs,sizeoftrain);
fprintf("guessed: %d/%d rate: %.2f%%\n", guessed, sizeoftest,(guessed/sizeoftest) * 100);
fprintf("total error: %.2f\n", error);

%display_network(images(:,1:100));
out = net;

%print execution time
toc

end