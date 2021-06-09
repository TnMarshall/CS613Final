# CS613Final

Instructions for Obtaining Results Discussed in Paper:
1.Run the preprocessing.m MATLAB script. This will produce the feature reduction datasets and place them in the featureRedProcessed directory.
2.Run the bash script runAndStoreFeatureReduction.sh . This script will run the necesary code obtain the accuracy values for the decision tree, naive bayes, KNN, logistic regression, and voting ensamble classifiers
3.Run baggingTester.py and adaboost_1_tester.py . This will produce the accuracies for the bagging and adaboost classifiers
4.The accuracies are stored in a non-csv format in featureRedOutput.txt, and baggingTester.py and adaboost_1_tester.py write to the terminal output.
5.The data above should be copy and pasted and format matched to the following CSV files
5a. featureRedOutput.txt should be split into accuraciesRaceDataEx.csv and accuraciesRaceDataInc.csv. This is accomplished by copying and pasting the data under the corresponding race included and excluded datasets into their respective files. Then, replace all appearances of "])," with a newline, remove all appearances of " ", "[", "]", "(", and ")". The files should now be in csv format. Add the first row "decision_tree,naive_bayes,KNN,logistic_regression,voting_ensamble" to the beginning of both files.
5b. Follow the same steps as 5a except copying the data from the terminal and into the respective csv files for bagging and boosting. (baggingAccuraciesEx.csv, baggingAccuraciesInc.csv, adaboostAccuraciesEx.csv, and adaboostAccuraciesInc.csv) The first row for these files will be "bagging" and "adaboost" respectively.
6. Run the matlab script combineBaggingAndRest.m. This script combines the CSV files into two CSV files, one for each dataset.
7. Run the matlab script accuracyGraphing.m. This script produces the final accuracies graphs which demonstrate the performance of the classifiers.

8. To obtain the confusion matrix disccused in the results section, run the logisticRegression.py file. The confusion matrix will be printed to the terminal where the axes and classes correspond to those in the paper.
