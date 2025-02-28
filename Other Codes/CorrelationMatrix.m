% Load the Cleveland dataset
data = readtable('processed.cleveland.data', 'FileType', 'text', 'Delimiter', ',', 'ReadVariableNames', false);

% Add column names
data.Properties.VariableNames = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', ...
                                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'};

%removing any missing data that is in the data set

data = standardizeMissing(data, '?');
data = rmmissing(data);

data.target = data.target > 0;


% Select only the numerical columns
numericalData = data{:, {'age', 'trestbps', 'chol', 'thalach','exang','target'}};  % Select columns of interest

% Compute the correlation matrix
correlationMatrix = corr(numericalData);

% Display the correlation matrix
disp(correlationMatrix);


%REMEMBER
%A correlation coefficient close to 1 means the variables are positively correlated.
%A coefficient close to -1 means they are negatively correlated.
%A coefficient close to 0 means there is little or no linear relationship between the variables.

hold of;

xvalues = {'age', 'trestbps', 'chol', 'thalach','exang','target'};
yvalues = {'age', 'trestbps', 'chol', 'thalach','exang','target'};

h = heatmap(xvalues,yvalues,correlationMatrix);
h.Colormap = jet;
h.ColorLimits = [-1, 1]; % Set color range for correlation