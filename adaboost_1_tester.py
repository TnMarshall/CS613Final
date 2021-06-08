import time
from adaboost_1_callable import adaboost_1

accuraciesRaceEx = []
startEx = time.time()

for i in range(1,92):   #1-91 range doesn't include the last number
    print(i)
    fileName = "./featureRedProcessed/raceExcludedProcessedFeatures_" + str(i) + ".csv"
    # print(fileName)
    accuraciesRaceEx.append(adaboost_1(fileName))
elapsedEx = time.time() - startEx
print("Execution Time of Excluded: " + str(elapsedEx))

accuraciesRaceInc = []

startInc = time.time()
for i in range(1,102):   #1-91 range doesn't include the last number
    print(i)
    fileName = "./featureRedProcessed/raceIncludedProcessedFeatures_" + str(i) + ".csv"
    # print(fileName)
    accuraciesRaceInc.append(adaboost_1(fileName))
elapsedInc = time.time() - startInc

print("Execution Time of Included: " + str(elapsedEx))

print("Accuracies Race Data Excluded: ")
print(accuraciesRaceEx)
print("Accuracies Race Data Included: ")
print(accuraciesRaceInc)