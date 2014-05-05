function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

% Compute prediction matrix for each class 
h = theta*data;
% Prevent overflow subtracting maximum theta from each of theta terms before computing the exponential
h = bsxfun(@minus, h, max(h, [], 1));
% Precompute the exponential for each term 
h = exp(h);
% Compute the propability matrix for each class
h = bsxfun(@rdivide, h, sum(h));

hmax = zeros(1,size(h, 2));
[hmax, pred] = max(h);

% ---------------------------------------------------------------------

end

