clear; close all; clc

%% Setup

if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end
py.importlib.import_module('votingCallable')

% accuraciesRet = py.votingCallable.voting('processedData/raceIncludedProcessedFeatures.csv')