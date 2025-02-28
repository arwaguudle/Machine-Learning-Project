%This is the main code, combining the other codes that i have done into one
%big code
%%
% ill be using the heart disease data set (more specifically the cleveland
% process clean data) from UCI Machine Learning Repository



%in this section of the code, i will be providing the displying the
%statistical values of the data

%loading the data
data = readtable('processed.cleveland.data', 'FileType', 'text', 'Delimiter', ',', 'ReadVariableNames', false);


% Add column names (if not already present)
data.Properties.VariableNames = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', ...
                                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'};
%removing any missing data that is in the data set

data = standardizeMissing(data, '?');
data = rmmissing(data);

%making the target column binary (1/0, 1 if they have a heart diease and 0
%if they dont) for the target that are 1,2,3,4 for any indictation precense
%of heart disease

data.target = data.target > 0;
%just checking if the target data is just binary 
disp("new target data"),data.target;

%finding the mean,std, min and max (i have picked specific features to
%which i believe will have somewhat a strong correlation to the target)
needed_data = data{:, {'age', 'chol', 'trestbps', 'thalach'}};

%calculating the mean, standard deviation, minimum and maximum of these
%features
meanvalues = mean(needed_data);
stdvalues = std(needed_data);
minvalues = min(needed_data);
maxvalues = max(needed_data);

%creating a table of the result values that i have found.
results=[meanvalues',stdvalues',minvalues',maxvalues'];
resultsTable = array2table(results, 'VariableNames', {'Mean', 'Std', 'Min', 'Max'}, ...
                            'RowNames', {'Age', 'Cholesterol', 'Resting Blood Pressure', 'Max Heart Rate'});

% Displaying the results table
disp('Statistics for Selected Features:');
disp(resultsTable);
%%
%in this section of the code i will displaying some histograms for the
%specific features that i have previously picked to visualised the number
%of patients with that feauture and the patients with heart disease with
%that features


heartdisease = data(data.target > 0, {'age', 'sex', 'trestbps','chol','thalach'}); %using these specific vairables for the histograms 

%starting off with making a histogram for the genderof patients
%we start off by indicating 1 for males and 0 for female as presented in
%the data set
males = sum(data.sex == 1);
females = sum(data.sex == 0);

figure;

%making my axis as male and females
xaxis = ["male","female"];
yaxis = [males, females];
bar(xaxis,yaxis,'FaceColor','b','FaceAlpha',0.5); %using FaceAlpha for the transperacy


%on the same axes, im making another histogram for the gender of patients
%with heart disease
hold on;
hdmales = sum(heartdisease.sex == 1);
hdfemales = sum(heartdisease.sex == 0);
hdyaxis = [hdmales, hdfemales];
bar(xaxis,hdyaxis,'FaceColor','r','FaceAlpha',0.5);

%giving labels as a way to differentiate 
legend('All Individuals', 'Heart Disease', 'Location', 'Best');
%providing a title and labels for the axes

title('Histogram of gender');
ylabel('Frequency');
xlabel('Gender')

%making another histogram but on the ages of all patients vs the patients
%with heart disease
figure;
%all patients
histogram(data.age, 'FaceColor', 'b','FaceAlpha', 0.5);  
hold on;
%patients with heart disease
histogram(heartdisease.age,'FaceColor', 'r','FaceAlpha', 0.5);

%For the Age Histogram, the Y-axis will show how many individuals within each specified age range (bin).
%The X-axis is the ages range
%providing a title, and labelings
title('Histogram of Age: All Individuals vs Heart Disease');
xlabel('Age');
ylabel('Frequency');
legend('All Individuals', 'Heart Disease', 'Location', 'Best');


% Histogram for the cholesterol column of all patients, and the patients
% with heart disease on the same axes
figure;
%all patients
histogram(data.chol,'FaceColor', 'b','FaceAlpha', 0.5);
hold on;
%patients with heart disease
histogram(heartdisease.chol,'FaceColor', 'r','FaceAlpha', 0.5);

%labellings andn titles
title("Histogram of cholesterol values");
xlabel("cholesterol values");
ylabel("frequency");
legend('All Individuals', 'Heart Disease', 'Location', 'Best');

%For the Cholesterol Histogram, the Y-axis will show how many individuals in the dataset have a cholesterol level within each specified range (bin).
%The X-axis represents the cholesterol values themselves (e.g., cholesterol levels in mg/dL).

%Histogram for the blood pressure column of all patients vs the patients
%with heart disease on the same axes
figure;
%all patients
histogram(data.trestbps, 'FaceColor', 'b','FaceAlpha', 0.5);
hold on;
%patients with heart disease
histogram(heartdisease.trestbps,'FaceColor', 'r','FaceAlpha', 0.5);

%titles and labellings
title("Histogram of resting blood prssure");
xlabel("resting blood pressure values");
ylabel("frequency");
legend('All Individuals', 'Heart Disease', 'Location', 'northeast');


%Histogram for the maximum heart rate for every patients and the patients
%for heart disease on the same axes
figure;
%all patients
histogram(data.thalach,'FaceColor', 'b','FaceAlpha', 0.5);
hold on;
%patients with heart disease
histogram(heartdisease.thalach, 'FaceColor', 'r','FaceAlpha', 0.5);
title("Histogram of maximum heart rate");
xlabel("maximum heart rate values");
ylabel("frequency");

legend('All Individuals', 'Heart Disease', 'Location', 'northwest');
figure;
%%
%in this section of the code: i will be displaying a correlation matrix to
%see if there's any relationship between any of the features and the
%target


check_correlation_data = data{:, {'age', 'trestbps', 'chol', 'thalach','fbs','target'}};  %using these features to see if theres any correlation between them
% !!! they are the numerical data, for the correlation matrix to work

% Computing the correlation matrix and then displaying it
correlationMatrix = corr(check_correlation_data);
disp(correlationMatrix);

%REMINDER
%A correlation coefficient close to 1 means the variables are positively correlated.
%A coefficient close to -1 means they are negatively correlated.
%A coefficient close to 0 means there is little or no linear relationship between the variables.


%displaying the correlation matrix in a heatmap 
hold of;

xvalues = {'age', 'trestbps', 'chol', 'thalach','fbs', 'target'};
yvalues = {'age', 'trestbps', 'chol', 'thalach','fbs', 'target'};

h = heatmap(xvalues,yvalues,correlationMatrix);
h.Colormap = jet;
h.ColorLimits = [-1, 1]; % Giving a color range for correlation

%%
%in this section of the code, i will be splitting up the dataset 80:20
%training and testing, and then applying the n fold classifiction
%validation and then recording results and patterns

%ill also be using hyperparameters and feature selection

%                --LOGISTIC REGRESSION--
% Separate features (X) and target (y)
X = data{:, {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach'}};
y = data.target;
% Normalize the features
X = normalize(X);


% Split dataset into training and testing
% 80% training data 20% testing data
cv = cvpartition(data.target, 'HoldOut', 0.2); % 20% test data

%announcing the train, test for the features and target 
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);


%will be finding the best hyperparameter for the logistic regression


rng(2); %a random number generator to make sure that the results stay the same

% Define the grid of hyperparameters (Lambda values for regularization strength)
lambda_values = [0.0001,0.001,0.01,0.1,1];

% Initialize variables to store the best model and performance
best_accuracy = 0;
best_lambda = 0;


% Grid search: Loop over each Lambda value for Lasso regularization
for lambda = lambda_values
    
    % Train logistic regression model with Lasso regularization
    model_lr = fitclinear(X_train, y_train, 'Learner', 'logistic', ...
                          'Regularization', 'lasso', 'Lambda', best_lambda);
    
    % predicting the model
    predictions = predict(model_lr, X_test);

    
    % Calculating the accracy for each lamdba 
    accuracy = mean(predictions == y_test);

    
    % Track the best model (highest accuracy)
    if accuracy > best_accuracy
        best_accuracy = accuracy;
        best_lambda = lambda;
    end
end
disp(['Best Lambda: ', num2str(best_lambda)]);
disp(['Best Accuracy: ', num2str(best_accuracy)]);

%now that we have the best parameter: we can use it on out logistic
%regression

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

        % Train logistic regression model (L2 regularization)
        lr_model = fitclinear(X_train(trainIdx, :), y_train(trainIdx), 'Learner', 'logistic', ...
                              'Regularization', 'lasso', 'Lambda', best_lambda);

        % Predicting and getting the validation accuracy 
        predictions = (predict(lr_model, X_train(testIdx, :)));
        lr_validation_predictions = round(predictions); % Threshold instead of rounding % Round to 0 or 1 (binary)
        lr_validation_accuracy(i) = mean(lr_validation_predictions == y_train(testIdx));  %Calculating the validation accuracy of this

        training_predictions = round(predict(lr_model, X_train(trainIdx, :)));
        lr_training_accuracy(i) = mean(training_predictions == y_train(trainIdx)); %calculating the training accuracy for the logisitic regression

        true_positive = sum((lr_validation_predictions == 1) & (y_train(testIdx) == 1)); %Correctly predicted positives.
        true_negative = sum((lr_validation_predictions == 0) & (y_train(testIdx) == 0)); % Correctly predicted negatives.
        false_positive = sum((lr_validation_predictions == 1) & (y_train(testIdx) == 0)); % Incorrectly predicted positives.
        false_negative = sum((lr_validation_predictions == 0) & (y_train(testIdx) == 1)); % Incorrectly predicted negatives.

        lr_precision(i) = true_positive / (true_positive + false_positive);  % Precision = TP / (TP + FP)
        lr_recall(i) = true_positive / (true_positive + false_negative);     % Recall = TP / (TP + FN)
        lr_f1_scr(i) = (2 * (lr_precision(i) * lr_recall(i))) / (lr_precision(i) + lr_recall(i)); % F1 Score = 2 * (Precision * Recall ) / (Precision + Recall)

        %going to try to find the auc 
        [~, scores] = predict(lr_model, X_test); % Extract probabilities
        scores = scores(:, 2); % Keep the probabilities for the positive class (1)
        
        [x, y, ~, auc] = perfcurve(y_test, scores, 1);
        lr_auc = auc;

        %ending my time 
        lr_time(i)=toc;
end


%Producing the error
lr_error = 1- lr_validation_accuracy;
%%
%displaying the ROC curve for logistic regression
figure;
plot(x, y, 'LineWidth', 2,Color='black');
xlabel("False Positive Rate");
ylabel("True Positive Rate");
title(["ROC Curve for Logistic Regression (AUC = ", lr_auc, ""]);
grid off;

% making the confusion matrix forLogistic Regression
y_test = double(y_test);  %making catgegories the same type (numerical)
lr_validation_predictions = predict(lr_model, X_test);
lr_validation_predictions_binary = round(lr_validation_predictions); 

lr_confusion_matrix = confusionmat(y_test, lr_validation_predictions_binary);
disp("Confusion Matrix for Logistic Regression")
disp(lr_confusion_matrix);

%displaying the confusions matrices
figure;
confusionchart(y_test, lr_validation_predictions_binary, ...
    "Title", "Logistic Regression Confusion Matrix", ...
    "RowSummary", "row-normalized", ...
    "ColumnSummary", "column-normalized");
%%

%               --NAIVE BAYES--

% Separate features (X) and target (y)
X = data{:, {'age', 'sex', 'trestbps', 'chol', 'thalach'}};
y = data.target;
% Normalize the features
X = normalize(X);

% Split dataset into training and testing
cv = cvpartition(data.target, 'HoldOut', 0.2); % 20% test data

%announcing the train, test for the features and target 
trainingData = data(training(cv), :);
testingData = data(test(cv), :);

X_train = trainingData{:, 1:end-1};  % Features for training
y_train = trainingData{:, end};      % Target for training

X_test = testingData{:, 1:end-1};   % Features for testing
y_test = testingData{:, end};       % Target for testing


n = 10; % Number of folds
cv2 = cvpartition(y_train, 'KFold', n); % Creating the n-fold partition

%storing some  variables

nb_validation_accuracy = zeros(n,1); % stores the validation accuracy for every n, initially starting with 0
nb_training_accuracy = zeros(n, 1);
nb_precision = zeros(n, 1);         % storing the precision for every n
nb_recall = zeros(n, 1);            %  a vector that stores for every recall
nb_f1_scr = zeros(n, 1);          % storing the f1 scores 
nb_auc = zeros(n,1);
nb_time = zeros(n, 1); %storing the time for each of the folds 




    for i = 1:n
        % Training and validation data for the i-th fold
        trainIdx = training(cv2, i); % Indices for training data
        testIdx = test(cv2, i); % Indices for validation data

        % Start timing
        tic;
        
        % Define a smoothing parameter (tune this based on your dataset)
        smoothing_value = 3;
        nb_model = fitcnb(X_train(trainIdx, :), y_train(trainIdx), ...
                  'DistributionNames', 'kernel', ...
                  'Width', smoothing_value);  % Apply kernel smoothing


        % Validation predictions
        predictions = predict(nb_model, X_train(testIdx, :));
        nb_validation_predictions = round(predictions); % Round predictions to binary values
        nb_validation_accuracy(i) = mean(nb_validation_predictions == y_train(testIdx)); % Validation accuracy

        % Training predictions
        training_predictions = round(predict(nb_model, X_train(trainIdx, :)));
        nb_training_accuracy(i) = mean(training_predictions == y_train(trainIdx)); % Training accuracy

        %calculating for the precision, recall, and f1 Score using
        %TP,FP,TN,FN


        true_positive = sum((nb_validation_predictions == 1) & (y_train(testIdx) == 1)); %Correctly predicted positives.
        true_negative = sum((nb_validation_predictions == 0) & (y_train(testIdx) == 0)); % Correctly predicted negatives.
        false_positive = sum((nb_validation_predictions == 1) & (y_train(testIdx) == 0));% Incorrectly predicted positives.
        false_negative = sum((nb_validation_predictions == 0) & (y_train(testIdx) == 1)); % Incorrectly predicted negatives.

        nb_precision(i) = true_positive / (true_positive + false_positive);  % Precision = TP / (TP + FP)
        nb_recall(i) = true_positive / (true_positive + false_negative);     % Recall = TP / (TP + FN)
        nb_f1_scr(i) = 2 * (lr_precision(i) * lr_recall(i)) / (lr_precision(i) + lr_recall(i)); % F1-Score = 2 * (Precision * Recall ) / (Precision + Recall)
        
        [~, scores] = predict(nb_model, X_test); % Extract probabilities
        scores = scores(:, 2); % Keep the probabilities for the positive class (1)
        
        [x, y, ~, auc] = perfcurve(y_test, scores, 1);
        nb_auc = auc;

    
        % End timing
        nb_time(i) = toc;
    end


%Producing the error
nb_error = 1- nb_validation_accuracy;

%%
%displaying the ROC curve for naive bayes
figure;
plot(x, y, 'LineWidth', 2,Color='blue');
xlabel("False Positive Rate");
ylabel("True Positive Rate");
title(["ROC Curve for Naive Bayes (AUC = ", nb_auc, ""]);
grid off;

% makking the confusion matrix for Naive Bayes 
y_test = double(y_test); %making catgegories the same type (numerical)
nb_validation_predictions = predict(nb_model, X_test);
nb_validation_predictions_binary = round(nb_validation_predictions); 

nb_confusion_matrix = confusionmat(y_test, nb_validation_predictions_binary);
disp("Confusion Matrix for Naive Bayes")
disp(nb_confusion_matrix);


%now for the naive bayes
figure;
confusionchart(y_test, nb_validation_predictions_binary, ...
    "Title", "Naive Bayes Confusion Matrix", ...
    "RowSummary", "row-normalized", ...
    "ColumnSummary", "column-normalized");
%%

% Setting up a table of all of the results
results = [mean(lr_auc),mean(lr_validation_accuracy), mean(lr_training_accuracy), mean(lr_error),mean(lr_precision),mean(lr_recall),mean(lr_f1_scr), mean(lr_time);   % Logistic Regression
           mean(nb_auc),mean(nb_validation_accuracy), mean(nb_training_accuracy) ,mean(nb_error),mean(nb_precision),mean(nb_recall),mean(nb_f1_scr), mean(nb_time)];  % Naive Bayes


% convert results into a table
resultsTable = array2table(results, ...
    'VariableNames', {'AUC',' Validation Accuracy','Training Accuracy' 'Error','Precision','Recall','F1 Score', 'Time'}, ...
    'RowNames', {'Logistic Regression', 'Naive Bayes'});

% displaying the table
disp(resultsTable);


