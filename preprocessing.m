clear;close all; clc

data = readtable('communities.data', 'FileType', 'text'); 

varData = readtable('attributeNamesExtracted.txt', 'FileType', 'text');

varNames = varData{:,2};

data.Properties.VariableNames = varNames;

features = data(:,1:127);
violentCrimesPerPop = data{:,128};
% Incompletes:
% 2,3,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,122,123,124,125,127
% Race Data:
% 8,9,10,11,27,28,29,30,31,32

% 67 speak english well?

% unneeded: 4 (state name), 

dataFilteredRaceIncluded = features(:,[1,5:101,119,120,121]);
dataFilteredRaceExcluded = features(:,[1,5:7,12:26,33:101,119,120,121]);

dataFilteredRaceIncludedNumeric = dataFilteredRaceIncluded{:,:};
dataFilteredRaceExcludedNumeric = dataFilteredRaceExcluded{:,:};

%% Find feature correlation coefficients

corrCoefficientsExcluded = corrcoef([dataFilteredRaceExcludedNumeric, violentCrimesPerPop]);
corrToViolentEx = corrCoefficientsExcluded(1:end-1,end);

corrCoefficientsIncluded = corrcoef([dataFilteredRaceIncludedNumeric, violentCrimesPerPop]);
corrToViolentInc = corrCoefficientsIncluded(1:end-1,end);

[sortedCoefEx, indsEx] = sort(corrToViolentEx, 'descend');
[sortedCoefInc, indsInc] = sort(corrToViolentInc, 'descend');

[absSortedCoefEx, absIndsEx] = sort(abs(corrToViolentEx), 'descend');
[absSortedCoefInc, absIndsInc] = sort(abs(corrToViolentInc), 'descend');

sortedFeaturesExNumeric = dataFilteredRaceExcludedNumeric(:,indsEx);
sortedFeaturesIncNumeric = dataFilteredRaceIncludedNumeric(:,indsInc);

sortedFeaturesEx = dataFilteredRaceExcluded(:,indsEx);
sortedFeaturesInc = dataFilteredRaceIncluded(:,indsInc);

sortedFeaturesExAbs = dataFilteredRaceExcluded(:,absIndsEx);
sortedFeaturesIncAbs = dataFilteredRaceIncluded(:,absIndsInc);

%% Choose N features with most absolute correlation
N = -1;
if N == -1
    chosenFeaturesEx = sortedFeaturesExAbs(:,1:end);
    chosenFeaturesInc = sortedFeaturesIncAbs(:,1:end);
else
    chosenFeaturesEx = sortedFeaturesExAbs(:,1:N);
    chosenFeaturesInc = sortedFeaturesIncAbs(:,1:N); 
end

%% Combine Chosen Features with crime data


violentCrimeBinned = round(data{:,'ViolentCrimesPerPop'},1)*10;
violentCrimeBinned = array2table(violentCrimeBinned);
violentCrimeBinned.Properties.VariableNames = "ViolentCrimesPerPop";

% chosenRecombEx = [chosenFeaturesEx(:,:),data(:,128)];
% chosenRecombInc = [chosenFeaturesInc(:,:),data(:,128)];

chosenRecombEx = [chosenFeaturesEx(:,:),violentCrimeBinned];
chosenRecombInc = [chosenFeaturesInc(:,:),violentCrimeBinned];

%% Randomize observation order

rng(0)
randIndsEx = randperm(size(chosenRecombEx,1));
randIndsInc = randperm(size(chosenRecombInc,1));

chosenRecombExRand = chosenRecombEx(randIndsEx,:);
chosenRecombIncRand = chosenRecombInc(randIndsInc,:);

%% Export data to csv

writetable(chosenRecombExRand, 'processedData/raceExcludedProcessedFeatures.csv');
writetable(chosenRecombIncRand, 'processedData/raceIncludedProcessedFeatures.csv');