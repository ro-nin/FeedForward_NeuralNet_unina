function [guessed, error] = testnet(net, testSet, targetSet, errorFnc)
%TODO: DELETE
sizeoftest = size(testSet);
sizeoftest = sizeoftest(2);

error = 0;
guessed = 0;

for i = 1: sizeoftest    
    test = propagate(net, testSet(:, i)');    
    [val, idx] = max(test{end});   
    if(idx == find(targetSet(i, :)))
        guessed = guessed + 1;
    end
    
    error = errorFnc(test{end}, targetSet(i,:)');
end

end

