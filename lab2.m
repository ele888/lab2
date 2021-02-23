%% LAB 2 ELE888 - Linear Discriminant Functions
% Preprocess Data
clear
load irisdata.mat
%% extract unique labels (class names)
labels = unique(irisdata_labels);

%% generate numeric labels
numericLabels = zeros(size(irisdata_features,1),1);
for i = 1:size(labels,1)
    numericLabels(find(strcmp(labels{i},irisdata_labels)),:)= i;
end

%% build training data set for three class comparison
% merge feature samples with numeric labels for three class comparison (Iris
% Setosa vs. Iris Veriscolour vs. Iris Virginia
trainingSet = [irisdata_features(1:150,:) numericLabels(1:150,1)];

%% create data set A,B,C
%dataSetA Iris Setosa with features x2 and x3
dataSetA = trainingSet(1:50,2:3);
%dataSetB Iris Versicolor with features x2 and x3
dataSetB = trainingSet(51:100,2:3);
%dataSetC Iris Virginia with features x2 and x3
dataSetC = trainingSet(101:150,2:3);
%% New Data set from A,B
newDataSet = [dataSetA;dataSetB];
%% Prepare Data: Augmentation & Normalization (for feature vector)
% Augmentation 
nRows = size(newDataSet, 1); 
x0 = [ones(1,nRows)]';
x1x2 = newDataSet;
C1 = [ones(1,(nRows/2))]';
C2 = 2*[ones(1,nRows/2)]';
y = [C1;C2];
z = [x0 x1x2 y];
% Normalization
zHat = [z(1:(nRows/2),1:end); -1*z(((nRows/2)+1):end,1:end-1), C2];
%% Separate the data
% where 30% is training
% where 70% is testing
% Cross varidation (train: 30%, test: 70%)
cv = cvpartition(size(zHat,1),'HoldOut',0.7);
idx = cv.test;
% Separate to training and test data
dataTrain = [zHat(~idx,:)]';
dataTest  = [zHat(idx,:)]';
%% Compute the Weight Vector (Gradient Descent Alg.)
% Gradient Descetnt Alg. - initialize criterion
eta = 0.01;             % learning rate
theta = 0; 
a = [0 0 1]';           %augmented weight vector
k = 0; 
maxitr = 300; itr = 0; cdn = true;
set = dataTrain(1:end-1,:);

%call gradient descent functions
gradient = GJp(set) ;       % Gradient Jp
%perceptron = Jp(a,gradient)     % Jp(a) = sum of all misclassified (-a.T*y)
a = a';
miss_y= a*set ;          %use to find negative values = misclassifed values
index = miss_y < 0;     % create logical index
neg = miss_y(index) ;    % obtain all negative values

%perceptron = Newa - eta.*gradient
%% create feature plot with descision boundary
% decision boundary threshold
% plot
%% Gradient Function
% Gradient Jp
function gradient = GJp(set)
    S = sum(set,2);
    gradient = S*(-1);
end

%% Perceptron Criterion Function -error here
% Jp(a) = sum of all misclassified (-a.T*y)
function perceptron = Jp(a,gradient)
    a= a*set
    perceptron = a - eta*gradient
end

%% condition to stop iterations
%n*perceptron < theta
function cdn = condition(eta,perceptron)
    cdn = eta*perceptron;
end
%% Gradient Descent function
 
function [k, itr, a] = GD(k,eta,theta)

    while itr < maxitr
    k = k + 1;
    a = @Jp;
    itr = itr + 1;
    cdn = @condition;
    cdn < theta;
    end
    
printf('Number of iterations: %d \n', itr)
printf('Final value of k: %d \n', k)
printf('Final value of augmented weight vector, a: %d \n', a)
    end