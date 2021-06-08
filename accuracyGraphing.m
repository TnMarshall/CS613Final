clear; close all; clc

%% Load files

accuraciesInc = readtable('accuraciesFullInc.csv');
accuraciesEx = readtable('accuraciesFullEx.csv');


accIncNum = accuraciesInc{:,:}*100;
accExNum = accuraciesEx{:,:}*100;

%% Plot Accuracies
% lineColors = ['r','g','b','m','c','k','w'];
lineColors = ["#fc0303";"#1cfc03";"#0328fc";"#9d03fc";"#03e3fc";"#fc8003";"#000000"];
figure(1)
hold on
for i = 1:size(accIncNum,2)
    plot(accIncNum(:,i), 'Color', lineColors(i))
end
title("Accuracies With Varying Numbers of Features For Race Included Dataset")
xlabel("Number of Features")
ylabel("Percent Accuracy")
legend("Decision Tree", "Naive Bayes", "KNN", "Logistic Regression", "Voting Ensamble","Bagging", "Adaboost")

figure(2)
hold on
for i = 1:size(accExNum,2)
    plot(accExNum(:,i), 'Color', lineColors(i))
end
title("Accuracies With Varying Numbers of Features For Race Excluded Dataset")
xlabel("Number of Features")
ylabel("Percent Accuracy")
legend("Decision Tree", "Naive Bayes", "KNN", "Logistic Regression", "Voting Ensamble","Bagging", "Adaboost")

% figure(3)

figure(3)
hold on
for i = 1:size(accExNum,2)
    plot(accIncNum(1:end-1,i), 'o-', 'Color', lineColors(i))
    plot(accExNum(:,i), 'Color', lineColors(i))
end
title("Accuracies With Varying Numbers of Features For Race Included and Excluded Datasets")
xlabel("Number of Features")
ylabel("Percent Accuracy")
legend("Decision Tree Race Included", "Decision Tree Race Excluded", "Naive Bayes Race Included", "Naive Bayes Race Excluded", "KNN Race Included", "KNN Race Excluded", "Logistic Regression Race Included", "Logistic Regression Race Excluded", "Voting Ensamble Race Included", "Voting Ensamble Race Excluded", "Bagging Race Included", "Bagging Race Excluded", "Adaboost Race Included", "Adaboost Race Excluded")

%% Graph of Simple Comparison Between DT and LR

figure(4)
hold on
for i = [1,4]
    plot(accIncNum(:,i), 'o-', 'Color', lineColors(i))
    plot(accExNum(:,i), 'Color', lineColors(i))
end
title("Comparison of Decision Tree and Logistic Regression Classifier Accuracy")
xlabel("Number of Features")
ylabel("Percent Accuracy")
legend("Decision Tree Race Included", "Decision Tree Race Excluded", "Logistic Regression Race Included", "Logistic Regression Race Excluded")

%% Individual Classifier Graphs
classifierNames = ["Decision Tree Accuracy"; "Naive Bayes Accuracy"; "KNN Accuracy"; "Logistic Regression Accuracy"; "Voting Ensemble Accuracy"; "Bagging Accuracy"; "Adaboost"];
for i = 1:size(accExNum,2)
    figNum = 4+i;
    figure(figNum);
    hold on
    plot(accIncNum(:,i), 'r-')
    plot(accExNum(:,i), 'b-')
    xlabel("Number of Features")
    ylabel("Percent Accuracy")
    legend("Race Included", "Race Excluded")
    title(classifierNames(i))
end