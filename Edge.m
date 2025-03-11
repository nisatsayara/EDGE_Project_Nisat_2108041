% Step 1: Load the Dataset
digitDatasetPath = fullfile(matlabroot, 'toolbox', 'nnet', 'nndemos', 'nndatasets', 'DigitDataset');
imds = imageDatastore(digitDatasetPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Step 2: Preprocess the Data
% Split the dataset into training and testing sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.7, 'randomized');

% Resize images to 28x28
inputSize = [28 28 1];
imdsTrain = augmentedImageDatastore(inputSize, imdsTrain);
imdsTest = augmentedImageDatastore(inputSize, imdsTest);

% Step 3: Define the Machine Learning Model (CNN)
layers = [
    imageInputLayer([28 28 1]) % Input layer for 28x28 grayscale images
    convolution2dLayer(3, 8, 'Padding', 'same') % Convolution layer with 8 filters
    batchNormalizationLayer % Normalize activations
    reluLayer % Activation function
    maxPooling2dLayer(2, 'Stride', 2) % Max pooling layer
    convolution2dLayer(3, 16, 'Padding', 'same') % Convolution layer with 16 filters
    batchNormalizationLayer % Normalize activations
    reluLayer % Activation function
    fullyConnectedLayer(10) % Fully connected layer for 10 classes (digits 0-9)
    softmaxLayer % Softmax layer for classification
    classificationLayer]; % Output layer

% Step 4: Train the Model
options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.01, ...
    'MaxEpochs', 10, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train the network
net = trainNetwork(imdsTrain, layers, options);

% Step 5: Test the Model
YPred = classify(net, imdsTest); % Predict labels for test data
YTest = imdsTest.Labels; % True labels

% Calculate accuracy
accuracy = sum(YPred == YTest) / numel(YTest);
disp(['Test Accuracy: ', num2str(accuracy)]);

% Step 6: Visualize Results
% Display some test images with their predicted labels
figure;
for i = 1:9
    subplot(3, 3, i);
    img = readimage(imdsTest, i); % Read the i-th test image
    imshow(img); % Display the image
    title(['Predicted: ', char(YPred(i))]); % Show predicted label
end