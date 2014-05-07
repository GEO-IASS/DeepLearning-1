function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% inputSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

M = size(data, 2);

% Creating cells to store input data and activations
depth = numel(stack);
a = cell(depth+1,1);

% Computing activations for each hidden layer (feedForwardAutoencoder.m)
a{1} = data;
for d = 1:depth
  a{d+1} = sigmoid(stack{d}.w * a{d} + repmat(stack{d}.b,1,M));
end

% Computing Softmax predictions (softmaxPredict.m)
h = softmaxTheta*a{depth+1};            % Compute prediction matrix for each class 
h = bsxfun(@minus, h, max(h, [], 1));   % Prevent overflow subtracting maximum theta from 
                                        % each of theta terms before computing the exponential
h = exp(h);                             % Precompute the exponential for each term 
h = bsxfun(@rdivide, h, sum(h));        % Compute the propability matrix for each class
hmax = zeros(1,size(h, 2));		% 'hmax' stores max probabilities found
[hmax, pred] = max(h);			% 'pred' stores index of rows (class id) with max probabilities


% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
