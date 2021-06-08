clear; close all; clc

%% Load files

accuraciesInc = readtable('accuraciesWoBoostInc.csv');
accuraciesEx = readtable('accuraciesWoBoostEx.csv');


accIncNum = accuraciesInc{:,:}*100;
accExNum = accuraciesEx{:,:}*100;

%% Plot Accuracies
lineColors = ['r','g','b','m','c','k'];
lineColorsFormatted = ["ro-";"go-";"bo-";"mo-";"co-";"ko-"];

figure(1)
hold on
for i = 1:size(accIncNum,2)
    plot(accIncNum(:,i), lineColors(i))
end
title("Accuracies With Varying Numbers of Features For Race Included Dataset")
xlabel("Number of Features")
ylabel("Percent Accuracy")
legend("Decision Tree", "Naive Bayes", "KNN", "Logistic Regression", "Voting Ensamble","Bagging")

figure(2)
hold on
for i = 1:size(accExNum,2)
    plot(accExNum(:,i), lineColors(i))
end
title("Accuracies With Varying Numbers of Features For Race Excluded Dataset")
xlabel("Number of Features")
ylabel("Percent Accuracy")
legend("Decision Tree", "Naive Bayes", "KNN", "Logistic Regression", "Voting Ensamble","Bagging")

% figure(3)

figure(3)
hold on
for i = 1:size(accExNum,2)
    plot(accIncNum(:,i), lineColorsFormatted(i))
    plot(accExNum(:,i), lineColors(i))
end
title("Accuracies With Varying Numbers of Features For Race Included and Excluded Datasets")
xlabel("Number of Features")
ylabel("Percent Accuracy")
legend("Decision Tree Race Included", "Decision Tree Race Excluded", "Naive Bayes Race Included", "Naive Bayes Race Excluded", "KNN Race Included", "KNN Race Excluded", "Logistic Regression Race Included", "Logistic Regression Race Excluded", "Voting Ensamble Race Included", "Voting Ensamble Race Excluded", "Bagging Race Included", "Bagging Race Excluded")

%% Graph of Simple Comparison Between DT and LR

figure(4)
hold on
for i = [1,4]
    plot(accIncNum(:,i), lineColorsFormatted(i))
    plot(accExNum(:,i), lineColors(i))
end
title("Comparison of Decision Tree and Logistic Regression Classifier Accuracy")
xlabel("Number of Features")
ylabel("Percent Accuracy")
legend("Decision Tree Race Included", "Decision Tree Race Excluded", "Logistic Regression Race Included", "Logistic Regression Race Excluded")