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

for i = 1:size(corrToViolentEx,1)
    if isnan(corrToViolentEx(i))
        corrToViolentEx(i) = 0;
    end
end

for i = 1:size(corrToViolentInc,1)
    if isnan(corrToViolentInc(i))
        corrToViolentInc(i) = 0;
    end
end

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

%% extract feature names to match with sorted correlations
ExNames = sortedFeaturesEx.Properties.VariableNames;
IncNames = sortedFeaturesInc.Properties.VariableNames;

exNamesAndCorr = array2table(sortedCoefEx');
incNamesAndCorr = array2table(sortedCoefInc');

exNamesAndCorr.Properties.VariableNames = ExNames;
incNamesAndCorr.Properties.VariableNames = IncNames;

%do the same for absolute values

ExNamesAbs = sortedFeaturesExAbs.Properties.VariableNames;
IncNamesAbs = sortedFeaturesIncAbs.Properties.VariableNames;

exNamesAndCorrAbs = array2table(absSortedCoefEx');
incNamesAndCorrAbs = array2table(absSortedCoefInc');

exNamesAndCorrAbs.Properties.VariableNames = ExNamesAbs;
incNamesAndCorrAbs.Properties.VariableNames = IncNamesAbs;

% export
writetable(rows2vars(exNamesAndCorr), 'exNamesAndCorr.csv');
writetable(rows2vars(exNamesAndCorrAbs), 'exNamesAndCorrAbs.csv');
writetable(rows2vars(incNamesAndCorr), 'incNamesAndCorr.csv');
writetable(rows2vars(incNamesAndCorrAbs), 'incNamesAndCorrAbs.csv');

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

% uncomment to generate initial dataset
% writetable(chosenRecombExRand, 'processedData/raceExcludedProcessedFeatures.csv');
% writetable(chosenRecombIncRand, 'processedData/raceIncludedProcessedFeatures.csv');

%% Export Data of Various Dimensions

% First export all 

recombAbsEx = [sortedFeaturesExAbs(:,:), violentCrimeBinned];
recombAbsInc = [sortedFeaturesIncAbs(:,:), violentCrimeBinned];

randIndsAbsEx = randperm(size(recombAbsEx,1));
randIndsAbsInc = randperm(size(recombAbsInc,1));

randAbsEx = recombAbsEx(randIndsAbsEx,:);
randAbsInc = recombAbsInc(randIndsAbsInc,:);

for N = 1:(size(randAbsEx, 2)-1)
%     size(randAbsEx(:,1:N))
    setExp = randAbsEx(:,[1:N,end]);
    setName = "featureRedProcessed/raceExcludedProcessedFeatures_" + num2str(N) + ".csv";
    writetable(setExp, setName);
end

for N = 1:(size(randAbsInc, 2)-1)
%     size(randAbsEx(:,1:N))
    setExp = randAbsInc(:,[1:N,end]);
    setName = "featureRedProcessed/raceIncludedProcessedFeatures_" + num2str(N) + ".csv";
    writetable(setExp, setName);
end

%% Get priors

priorsEx = zeros(1,11);
priorsInc = zeros(1,11);

for i = 1:size(randAbsEx,1)
    priorsEx(randAbsEx{i,end}+1) = priorsEx(randAbsEx{i,end}+1) + 1;
    priorsInc(randAbsInc{i,end}+1) = priorsInc(randAbsInc{i,end}+1) + 1;
end

priorsExPerc = priorsEx / sum(priorsEx);
priorsIncPerc = priorsInc / sum(priorsInc);