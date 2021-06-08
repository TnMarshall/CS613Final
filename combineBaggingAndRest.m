clear; close all; clc

%% Combine Bagging and Rest

accuraciesInc = readtable('accuraciesRaceDataInc.csv');
accuraciesEx = readtable('accuraciesRaceDataEx.csv');

baggingInc = readtable('baggingAccuraciesInc.csv');
baggingEx = readtable('baggingAccuraciesEx.csv');

adaboost1Inc = readtable('adaboostAccuraciesInc.csv');
adaboost1Ex = readtable('adaboostAccuraciesEx.csv');

fullInc = [accuraciesInc, baggingInc,adaboost1Inc];
fullEx = [accuraciesEx, baggingEx, adaboost1Ex];

writetable(fullInc, 'accuraciesFullInc.csv')
writetable(fullEx, 'accuraciesFullEx.csv')