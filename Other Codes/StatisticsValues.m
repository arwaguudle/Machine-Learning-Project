%ill be using the heart disease data set (more specifically the cleveland
%process clean data)

% Load and preprocess data
data = readtable('processed.cleveland.data', 'FileType', 'text', 'Delimiter', ',', 'ReadVariableNames', false);


% Add column names (if not already present)
data.Properties.VariableNames = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', ...
                                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'};
%removing any missing data that is in the data set

data = standardizeMissing(data, '?');
data = rmmissing(data);

%making the target column binary (1/0, 1 if they have a heart diease and 0
%if they dont)
data.target = data.target > 0;
%just checking if the target data is just binary 
display ("new target data"),data.target

%finding the mean,std, min and max
needed_data = data{:, {'age', 'chol', 'trestbps', 'thalach'}};


meanvalues = mean(needed_data);
stdvalues = std(needed_data);
minvalues = min(needed_data);
maxvalues = max(needed_data);

disp('Minimum Values:');
disp(minvalues);

disp('Maximum Values:');
disp(maxvalues);

disp('Mean Values:');
disp(meanvalues);

disp('Standard Deviation Values:');
disp(stdvalues);


results=[meanvalues',stdvalues',minvalues',maxvalues']

resultsTable = array2table(results, 'VariableNames', {'Mean', 'Std', 'Min', 'Max'}, ...
                            'RowNames', {'Age', 'Cholesterol', 'Resting Blood Pressure', 'Max Heart Rate'});

% Display the results table
disp('Statistics for Selected Features:');
disp(resultsTable);