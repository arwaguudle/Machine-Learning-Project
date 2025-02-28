% Load the Cleveland dataset
data = readtable('processed.cleveland.data', 'FileType', 'text', 'Delimiter', ',', 'ReadVariableNames', false);

% Add column names
data.Properties.VariableNames = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', ...
                                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'};


% Standardize missing values and remove rows with missing data
data = standardizeMissing(data, '?');
data = rmmissing(data);


% Convert the target column to binary (1 for heart disease, 0 for no heart disease)
data.target = data.target > 0;

% Separate features (X) and target (y)
X = data{:, {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach'}};
y = data.target;
%%
% Split dataset into training and testing
% 80% training data 20% testing data
cv = cvpartition(data.target, 'HoldOut', 0.2); % 20% test data

trainingData = data(training(cv), :);
testingData = data(test(cv), :);


X_train = trainingData{:, 1:end-1};  % Features for training
y_train = trainingData{:, end};      % Target for training

X_test = testingData{:, 1:end-1};   % Features for testing
y_test = testingData{:, end};       % Target for testing

%%
% Logisitc regression
%now to apply the 10 fold classificaition on the training set
n = 10; %number of folds

cv2 = cvpartition(y_train,'KFold', n); %creating the nfold partition

%storing some  variables
lr_validation_accuracy = zeros(n,1); % storing the validation accuracy for every n, initially starting with 0
lr_training_accuracy = zeros(n, 1);
lr_precision = zeros(n, 1);         % storing the precision for every n
lr_recall = zeros(n, 1);            %  a vector that stores for every recall
lr_f1_scr = zeros(n, 1);          % storing the f1 scores 
lr_auc = zeros(n,1); % a table which starts from 0 and then stores the auc for every n
lr_time = zeros(n, 1); %storing the time for each of the folds 


for i = 1:n
    % Training and validation data for the i-th fold
    trainIdx = training(cv2, i);
    testIdx = test(cv2, i);

    %starting up my time
    tic;

    % Train logistic regression model
    model_lr = fitglm(X_train(trainIdx, :), y_train(trainIdx), 'Distribution', 'binomial');
    
    % Predicting and getting the validation accuracy 
    predictions = (predict(model_lr, X_train(testIdx, :)));
    validation_predictions = round(predictions); % Round to 0 or 1 (binary)
    lr_validation_accuracy(i) = mean(validation_predictions == y_train(testIdx));  %Calculating the validation accuracy of this

    training_predictions = round(predict(model_lr, X_train(trainIdx, :)));
    lr_training_accuracy(i) = mean(training_predictions == y_train(trainIdx)); %calculating the training accuracy for the logisitic regression

    % Precision, Recall, and F1-Score
    true_positive = sum((validation_predictions == 1) & (y_train(testIdx) == 1));
    true_negative = sum((validation_predictions == 0) & (y_train(testIdx) == 0));
    false_positive = sum((validation_predictions == 1) & (y_train(testIdx) == 0));
    false_negative = sum((validation_predictions == 0) & (y_train(testIdx) == 1));

    lr_precision(i) = true_positive / (true_positive + false_positive);  % Precision = TP / (TP + FP)
    lr_recall(i) = true_positive / (true_positive + false_negative);     % Recall = TP / (TP + FN)
    lr_f1_scr(i) = 2 * (lr_precision(i) * lr_recall(i)) / (lr_precision(i) + lr_recall(i)); % F1-Score


    %Calculating the AUC (using the validation predictions
    %[X, Y, ~, auc] = perfcurve(y_train(testIdx), predictions(:, 1), 1);
    %lr_auc(i) = auc;
    

    %ending my time 
    lr_time(i)=toc;
end
%{
% Average Validation Accuracy of Logistic regression
disp("Logistic Regression Validation Accuracy:");
disp(mean(lr_validation_accuracy));

%Average Training Accuracy for Logisitc regression
disp("Logistic Regression Training Accuracy:");
disp(mean(lr_training_accuracy));

%Average AUC for Logisitc regression
disp("Logistic Regression AUC:");
disp(mean(lr_auc));

%Producing the error
lr_error = 1- lr_accuracy;
%Displaying the average error 
disp("Logistic Regression Error:")
disp(mean(lr_error));


%i also need to take into account how long the model has taken to build
disp("Logistic Regression Time:");
%disp(time);
disp(mean(lr_time))

%displaying the precicison, recall, and f1 score
disp("Logistic Regression Precision:")
disp(mean(lr_precision))

disp("Logistic Regression Recall:")
disp(mean(lr_recall))


disp("Logistic Regression F1 Score:")
disp(mean(lr_f1_scr))
%}
%%
%Naive Bayes

%now to apply the 10 fold classificaition on the training set
n = 10; %number of folds


%storing some  variables
cv2 = cvpartition(y_train,'KFold', n); %creating the nfold partition
nb_validation_accuracy = zeros(n,1); % stores the validation accuracy for every n, initially starting with 0
nb_training_accuracy = zeros(n, 1);
nb_precision = zeros(n, 1);         % storing the precision for every n
nb_recall = zeros(n, 1);            %  a vector that stores for every recall
nb_f1_scr = zeros(n, 1);          % storing the f1 scores 
nb_auc = zeros(n,1);
nb_time = zeros(n, 1); %storing the time for each of the folds 

for i = 1:n
    % Training and validation data for the i-th fold
    trainIdx = training(cv2, i);
    testIdx = test(cv2, i);

    %starting up my time
    tic;

    % Train naive bayes model
    model_nb = fitcnb(X_train(trainIdx, :), y_train(trainIdx));
    
    % finding the validation accuracy 
    predictions = predict(model_nb, X_train(testIdx, :));
    validation_predictions = round(predictions);
    
    nb_validation_accuracy(i) = mean(validation_predictions == y_train(testIdx));  %Calculating the validation accuracy for naive bayes
    
    %finding the training accuracy
    training_predictions = round(predict(model_nb, X_train(trainIdx, :)));
    nb_training_accuracy(i) = mean(training_predictions == y_train(trainIdx)); %calculating the training accuracy 

    % Precision, Recall, and F1-Score
    true_positive = sum((validation_predictions == 1) & (y_train(testIdx) == 1));
    true_negative = sum((validation_predictions == 0) & (y_train(testIdx) == 0));
    false_positive = sum((validation_predictions == 1) & (y_train(testIdx) == 0));
    false_negative = sum((validation_predictions == 0) & (y_train(testIdx) == 1));

    nb_precision(i) = true_positive / (true_positive + false_positive);  % Precision = TP / (TP + FP)
    nb_recall(i) = true_positive / (true_positive + false_negative);     % Recall = TP / (TP + FN)
    nb_f1_scr(i) = 2 * (lr_precision(i) * lr_recall(i)) / (lr_precision(i) + lr_recall(i)); % F1-Score

    %finding the auc for naive bayes
    %[X, Y, ~, auc] = perfcurve(y_train(testIdx), predictions, 1);
    %nb_auc(i) = auc;

    %ending my time 
    nb_time(i)=toc;
end

%{
% Average validation accuracy from k-fold
disp("Naive Bayes Validation Accuracy:");
disp(mean(nb_training_accuracy));


%calculating the training accuracy
disp("Naive Bayes Training Accuracy:");
disp(mean(nb_validation_accuracy));


%Producing the error
nb_error = 1- nb_accuracy;
%Displaying the average error 
disp("Naive Bayes Error")
disp(mean(nb_error));


%i also need to take into account how long the model has taken to build
disp("Naive Bayes Time");
disp(mean(nb_time))

%displaying the precicison, recall, and f1 score
disp("Naive Bayes Precision:")
disp(mean(nb_precision))

disp("Naive Bayes Recall:")
disp(mean(nb_recall))


disp("Naive Bayes F1 Sccore:")
disp(mean(nb_f1_scr))
%}


%%
% Setting up a table of results
results = [mean(lr_validation_accuracy), mean(lr_training_accuracy), mean(lr_error),mean(lr_precision),mean(lr_recall),mean(lr_f1_scr), mean(lr_time);   % Logistic Regression
           mean(nb_validation_accuracy), mean(nb_training_accuracy) ,mean(nb_error),mean(nb_precision),mean(nb_recall),mean(nb_f1_scr), mean(nb_time)];  % Naive Bayes

% Convert results into a table
resultsTable = array2table(results, ...
    'VariableNames', {' Validation Accuracy','Training Accuracy' 'Error','Precision','Recall','F1 Score', 'Time'}, ...
    'RowNames', {'Logistic Regression', 'Naive Bayes'});

% Display the table
disp(resultsTable);
