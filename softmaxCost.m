function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);   
[n m] = size(data);   
groundTruth = full(sparse(labels, 1:m, 1));   
cost = 0;  

thetagrad = zeros(numClasses, n);   

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

%{
for i = 1:m
    sum_prob = sum(exp(thetagrad*data(:,i)));  %Finding sum of probability of all classes
    cost = cost + log((exp(thetagrad(labels(i),:)*data(:,i)) / sum_prob));
    %Finding gradient of theta for each class.
    for j= 1:numClasses
        thetagrad(j,:) = thetagrad(j,:) + (groundTruth(j,i) - (exp(thetagrad(j,:)*data(:,i)) / sum_prob))';  
    end
end
   
cost = -(cost/m);% + sum(sum(theta .^ 2))*lambda/2;  
thetagrad = -thetagrad* data/m;% + lambda.*theta;   
%}

%%

[nfeatures, nsamples] = size(data);
M = theta * data;
maxM = max(M, [], 1);
M = bsxfun(@minus, M, maxM);
M = exp(M);
M = bsxfun(@rdivide, M, sum(M));
temp = groundTruth .* log(M);
cost = - sum(sum(temp)) ./ nsamples;
cost = cost + sum(sum(theta .^ 2)) .* lambda ./ 2;
 
temp = groundTruth - M;
temp = temp * data';
thetagrad = - temp ./ nsamples;
thetagrad = thetagrad + lambda .* theta;
%%

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = thetagrad(:);
end

