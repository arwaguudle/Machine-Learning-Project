%in this code, i will be experimenting different parameters and checking to
%see which one gives good results that we have found before in the training
%and testing sets
%%
% Load the Cleveland dataset
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
% Split dataset into training and testing
% 80% training data 20% testing data
cv = cvpartition(data.target, 'HoldOut', 0.2); % 20% test data

% Split dataset into training and testing (80% train, 20% test)
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

%%
% Standardize the training features
mean_X_train = mean(X_train);
std_X_train = std(X_train);
X_train_standardized = (X_train - mean_X_train) ./ std_X_train;
X_test_standardized = (X_test - mean_X_train) ./ std_X_train;  % Use training mean and std

%%


% Define the grid of hyperparameters (Lambda values for regularization strength)
lambda_values = [0.001, 0.01, 0.1, 1, 10, 100];

% Initialize variables to store the best model and performance
best_accuracy = 0;
best_lambda = 0;
results = [];  % To store the results

% Grid search: Loop over each Lambda value for Lasso regularization
for lambda = lambda_values
    % Train logistic regression model with Lasso regularization
    model_lr = fitclinear(X_train_standardized, y_train, 'Learner', 'logistic', ...
                          'Regularization', 'lasso', 'Lambda', lambda);
    
    % Predict on the test set
    predictions = predict(model_lr, X_test_standardized);
    
    % Calculate accuracy
    accuracy = mean(predictions == y_test);
    
    % Store the results
    results = [results; lambda, accuracy];
    
    % Track the best model (highest accuracy)
    if accuracy > best_accuracy
        best_accuracy = accuracy;
        best_lambda = lambda;
    end
end


disp('Lambda   Accuracy');
disp(results);

%displaying the best lambda and its accuracy
disp(best_lambda)
disp(best_accuracy)
%%
%combining it on the logistic regression on the last code:
n = 10; %number of folds


cv2 = cvpartition(y_train,'KFold', n); %creating the nfold partition
lr_accuracy = zeros(n,1); % a table where n are the number of rows and theres 1 column, it stores the accuracy for every n, initially starting with 0
lr_training_accuracy = zeros(n, 1);
lr_auc = zeros(n,1); % a table which starts from 0 and then stores the auc for every n
lr_time = zeros(n, 1); %storing the time for each of the folds 

lamba = best_lambda;
% For storing results
results = [];

for lamba = lamba

    for i = 1:n
        % Training and validation data for the i-th fold
        trainIdx = training(cv2, i);
        testIdx = test(cv2, i);

        %starting up my time
        tic;

        % Train logistic regression model (L2 regularization)
        model_lr = fitclinear(X_train(trainIdx, :), y_train(trainIdx), 'Learner', 'logistic', ...
                              'Regularization', 'ridge', 'Lambda', lambda);

        % Predicting and getting the validation accuracy 
        predictions = (predict(model_lr, X_train(testIdx, :)));
        validation_predictions = round(predictions); % Round to 0 or 1 (binary)
        lr_accuracy(i) = mean(validation_predictions == y_train(testIdx));  %Calculating the validation accuracy of this

        training_predictions = round(predict(model_lr, X_train(trainIdx, :)));
        lr_training_accuracy(i) = mean(training_predictions == y_train(trainIdx)); %calculating the training accuracy for the logisitic regression

        %Calculating the AUC (using the validation predictions
        %[X, Y, ~, auc] = perfcurve(y_train(testIdx), predictions(:, 1), 1);
        %lr_auc(i) = auc;
    

        %ending my time 
        lr_time(i)=toc;
    end 
end

% Average Validation Accuracy of Logistic regression
disp("Logistic Regression Validation Accuracy:");
disp(mean(lr_accuracy));

%Average Training Accuracy for Logisitc regression
disp("Logistic Regression Training Accuracy:");
disp(mean(lr_training_accuracy));

%Average AUC for Logisitc regression
disp("Logistic Regression AUC");
disp(mean(lr_auc));

%Producing the error
lr_error = 1- lr_accuracy;
%Displaying the average error 
disp("Logistic Regression Error")
disp(mean(lr_error));


%i also need to take into account how long the model has taken to build
disp("Time for each fold:");
%disp(time);
disp(mean(lr_time))