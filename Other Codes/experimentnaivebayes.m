%doing the same thing for naive bayes, i need to find a good hyperparameter
%for it 
%%
%loading the data 
data = readtable('processed.cleveland.data', 'FileType', 'text', 'Delimiter', ',', 'ReadVariableNames', false);

% Add column names
data.Properties.VariableNames = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', ...
                                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'};

data = standardizeMissing(data, '?');

% Convert the target column to binary (1 for heart disease, 0 for no heart disease)
data.target = data.target > 0;

% Separate features (X) and target (y)
X = data{:, {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach'}};
y = data.target;
%%

%doing for naive bayes now
% Split dataset into training and testing
cv = cvpartition(data.target, 'HoldOut', 0.2); % 20% test data

trainingData = data(training(cv), :);
testingData = data(test(cv), :);

X_train = trainingData{:, 1:end-1};  % Features for training
y_train = trainingData{:, end};      % Target for training

X_test = testingData{:, 1:end-1};   % Features for testing
y_test = testingData{:, end};       % Target for testing
%%
% Naive Bayes with 10-fold cross-validation for Gaussian distribution and smoothing

n = 10; % Number of folds
cv2 = cvpartition(y_train, 'KFold', n); % Creating the n-fold partition

% Initialize storage for metrics
nb_accuracy = zeros(n, 1); % Stores validation accuracy for each fold
nb_training_accuracy = zeros(n, 1); % Stores training accuracy for each fold
nb_auc = zeros(n, 1); % Stores AUC for each fold
nb_time = zeros(n, 1); % Stores computation time for each fold

% Define a smoothing parameter (tune this based on your dataset)
smoothing_value = 1e-2; % Example value, adjust if needed

for i = 1:n
    % Training and validation data for the i-th fold
    trainIdx = training(cv2, i); % Indices for training data
    testIdx = test(cv2, i); % Indices for validation data

    % Start timing
    tic;

    % Train Naive Bayes model with Gaussian (normal) distribution and smoothing
    model_nb = fitcnb(X_train(trainIdx, :), y_train(trainIdx), ...
                  'DistributionNames', 'normal'); % Gaussian Naive Bayes
    % Validation predictions
    predictions = predict(model_nb, X_train(testIdx, :));
    validation_predictions = round(predictions); % Round predictions to binary values
    nb_accuracy(i) = mean(validation_predictions == y_train(testIdx)); % Validation accuracy

    % Training predictions
    training_predictions = round(predict(model_nb, X_train(trainIdx, :)));
    nb_training_accuracy(i) = mean(training_predictions == y_train(trainIdx)); % Training accuracy

    % Compute AUC for validation
    %[X, Y, ~, auc] = perfcurve(y_train(testIdx), predictions, 1);
    %nb_auc(i) = auc;

    % End timing
    nb_time(i) = toc;
end

% Display results
disp("Naive Bayes Results (Gaussian Distribution with Smoothing):");

% Validation accuracy
disp("Average Validation Accuracy:");
disp(mean(nb_accuracy));

% Training accuracy
disp("Average Training Accuracy:");
disp(mean(nb_training_accuracy));

% Error rate
nb_error = 1 - nb_accuracy;
disp("Average Validation Error:");
disp(mean(nb_error));

% AUC
disp("Average AUC:");
disp(mean(nb_auc));

% Computation time
disp("Average Time for Each Fold:");
disp(mean(nb_time));
