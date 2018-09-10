
train_im = loadMNISTImages('train-images.idx3-ubyte');
train_lb = loadMNISTLabels('train-labels.idx1-ubyte');
train_lb = train_lb';

train_lb(train_lb==0) = 10;
train_lb = dummyvar(train_lb);

ts_size = 500;
k = 5;

slice_size = int32(ts_size / k);

resized_im = train_im(:, 1:ts_size);
resized_lb = train_lb(1:ts_size, :);

learning_rates = [1.e-1, 1.e-2, 1.e-3, 1.e-4, 1.e-5, 1.e-6];
nodes = [100, 200, 300, 400, 500];

result = cell(k, 1);

for i = 1: k
    
    start_idx = slice_size * (i-1) + 1;
    stop_idx = slice_size * (i-1) + slice_size;
    
    fprintf("%d - %d\n", start_idx, stop_idx);
    
    k_train_im = [resized_im(:, 1:start_idx), resized_im(:, stop_idx:ts_size)]; 
    k_train_lb = [resized_lb(1:start_idx, :); resized_lb(stop_idx:ts_size,:)]; 
    
    k_test_im = resized_im(:, start_idx:stop_idx); 
    k_test_lb = resized_im(start_idx:stop_idx, :);
    
    net = ..........;
    
    net = train();
    result{i} = test();        
    
end

    % pick the best
