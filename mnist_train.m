function out = mnist_train(net, epochs, eta, sizeoftrain, sizeoftest, batch_size)

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

if nargin < 6
    fprintf("not enough params, setting to default\n");
    epochs=1;
    sizeoftrain = 1000;
    sizeoftest = 10000;
    eta=0.1;
    % set online mode
    batch_size = 1;
end

%start counting execution time
tic
for i = 1: epochs

    for im = 1: batch_size: sizeoftrain
        end_im = im + batch_size - 1;
        net = train(net, train_im(:, im:end_im)', train_lb(im:end_im, :)', eta);
    end
    
    guessed = 0;
    error = 0;

    for j = 1: sizeoftest    
        test = propagate(net, test_im(:, j)');    
        [val, idx] = max(test{end});   
        if(idx == find(test_lb(j, :)))
            guessed = guessed + 1;
        end

        %calcolo dell'errore a seconda della funzione di costo associata
        if isequal(net.costFnc,@quadraticCost)
            error = error + 0.5 * (sum(test{end} - test_lb(j,:)')^2);
        elseif isequal(net.costFnc,@crossEntropy)
            error = error + sum( log(test{end}) .* test_lb(j,:)' );
        end
            
    end
    
    accuracy = (guessed/sizeoftest) * 100;
    fprintf("guessesed: %d/%d - accuracy: %.2f%%\n", guessed, sizeoftest, accuracy);
    fprintf("epoch: %d, total error: %f\n", i, error);

end
%print execution time
toc

out = net;

end