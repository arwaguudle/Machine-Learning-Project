% Load the Cleveland dataset
data = readtable('processed.cleveland.data', 'FileType', 'text', 'Delimiter', ',', 'ReadVariableNames', false);

% Add column names
data.Properties.VariableNames = {'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', ...
                                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'};

data = standardizeMissing(data, '?');
data = rmmissing(data);

% Filter the data for individuals with heart disease (target = 1,2,3,4)
heartdisease = data(data.target > 0, {'age', 'sex', 'cp', 'trestbps','chol','thalach'}); 

%%
males = sum(data.sex == 1);
females = sum(data.sex == 0);

xaxis = ["male","female"];
yaxis = [males, females];

figure;
bar(xaxis,yaxis,'FaceColor','b','FaceAlpha',0.5);


title('Histogram of gender(with and without heart diease)');
ylabel('Frequency');
xlabel('Gender')

hold on;
hdmales = sum(heartdisease.sex == 1);
hdfemales = sum(heartdisease.sex == 0);
hdyaxis = [hdmales, hdfemales];
bar(xaxis,hdyaxis,'FaceColor','r','FaceAlpha',0.5);

legend('All Individuals', 'Heart Disease', 'Location', 'Best');

%%
figure;
histogram(data.age, 'FaceColor', 'b','FaceAlpha', 0.5);  %using FaceAlpha for the transperacy
hold on;

% Plot the histogram for individuals with heart disease (red)
histogram(heartdisease.age,'FaceColor', 'r','FaceAlpha', 0.5);

title('Histogram of Age: All Individuals vs Heart Disease');
xlabel('Age');
ylabel('Frequency');

legend('All Individuals', 'Heart Disease', 'Location', 'Best');

%For the Age Histogram, the Y-axis will show how many individuals within each specified age range (bin).
%The X-axis the ages range


%%
% Histogram for the cholesterol column
figure;
histogram(data.chol,'FaceColor', 'b','FaceAlpha', 0.5);
hold on;
histogram(heartdisease.chol,'FaceColor', 'r','FaceAlpha', 0.5);
title("Histogram of cholesterol values");
xlabel("cholesterol values");
ylabel("frequency");

legend('All Individuals', 'Heart Disease', 'Location', 'Best');

%For the Cholesterol Histogram, the Y-axis will show how many individuals in the dataset have a cholesterol level within each specified range (bin).
%The X-axis represents the cholesterol values themselves (e.g., cholesterol levels in mg/dL).

%%
%Histogram for the blood pressure column
figure;
histogram(data.trestbps, 'FaceColor', 'b','FaceAlpha', 0.5);
hold on;
histogram(heartdisease.trestbps,'FaceColor', 'r','FaceAlpha', 0.5);
title("Histogram of resting blood prssure");
xlabel("resting blood pressure values");
ylabel("frequency");


legend('All Individuals', 'Heart Disease', 'Location', 'northeast');
%%
%Histogram for the maximum heart rate
figure;
histogram(data.thalach,'FaceColor', 'b','FaceAlpha', 0.5);
hold on;

histogram(heartdisease.thalach, 'FaceColor', 'r','FaceAlpha', 0.5);
title("Histogram of maximum heart rate");
xlabel("maximum heart rate values");
ylabel("frequency");

legend('All Individuals', 'Heart Disease', 'Location', 'northwest');
%%