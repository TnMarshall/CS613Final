clear; close all; clc

%% Combine Bagging and Rest

accuraciesInc = readtable('accuraciesRaceDataInc.csv');
accuraciesEx = readtable('accuraciesRaceDataEx.csv');

baggingInc = readtable('baggingAccuraciesInc.csv');
baggingEx = readtable('baggingAccuraciesEx.csv');

fullInc = [accuraciesInc, baggingInc];
fullEx = [accuraciesEx, baggingEx];

writetable(fullInc, 'accuraciesWoBoostInc.csv')
writetable(fullEx, 'accuraciesWoBoostEx.csv')