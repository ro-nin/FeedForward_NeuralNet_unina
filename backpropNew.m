function backprop(net, imagename, labelname)
    
    images = loadMNISTImages('train-images.idx3-ubyte');
    labels = loadMNISTLabels('train-labels.idx1-ubyte');
    
    % disp(images);
    % display_network(images(:,1:100)); % Show the first 100 images
    % disp(labels(1:10));
    
    N = size(images);
    N = N(2);
    
    % numero di strati della rete
    len = length(net.hiddenSize);
    
    targets = zeros(10, N);
  
    for i = 1: N
        targets(labels(i, 1)+1, i) = 1;
    end
        
    
    for i = 1: 1
       x = images(:, i);       
       delta = cell(len, 1);
       y = forwardpropagation(net, x);
       %TODO ADD BIAS
       % disp(y{len});
       % disp(targets(:,i))
       deltaKout = y{len} - targets(:,i);        
       % disp(deltaKout);
       
       %zHn = y(1:len-1);
       %OneMinuszHn = 1-zHn;
       %deltaKhidden = zHn .* OneMinuszHn;
       
       deltaKhidden = cell(len-2,1);
       for j = 1:len-2
           zHn=y{len-j};
           OneMinuszHn = 1-zHn;
           deltaKhidden{len-j-1}=zHn .* OneMinuszHn;
       end
       
       last=deltaKout;
       for j = 1:len-2
           currLen = len-j;
           wKh = net.weights{len-j};
           temp=0;
           for h = 1:net.hiddenSize(currLen)
                %column=wKh(1:net.hiddenSize(currLen+1),h);
                %disp(column);
                %temp=temp+ sum((column.*last));
                row=wKh(h,1:net.hiddenSize(currLen+1));
                temp=sum((row * last));
                disp(deltaKhidden{j}(h));
                deltaKhidden{j}(h)=deltaKhidden{j}(h) * temp;
           end
           last=deltaKhidden;
       end
       
    end
end







