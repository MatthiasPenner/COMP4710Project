import pandas as pd

df = pd.read_csv("Labels.csv")
cols = len(df.columns)
dict = {}
malicious = 0
isMalicious = 0
falsePositive = 0
falseNegative = 0


for i in range(len(df)):
    entry = 0

    #Check if entry is malicious from csv
    if(df.iloc[i,0] == 1):
        malicious += 1

    for j in range(1,cols):
        total = 0
        count = 0
        #Check if count for the interval already exists - else add to dictionary
        if (df.columns[j]+str(df.iloc[i,j])) in dict:
            count = dict.get(df.columns[j]+str(df.iloc[i,j]))
        else:
            count = df.value_counts(df.columns[j])[df.iloc[i,j]]
            dict.__setitem__(df.columns[j]+str(df.iloc[i,j]), count)

        #Check if count for interval AND Class==1 exists - else add to dictionary
        if(df.iloc[i,0] == 1):
            if (df.columns[j] + str(df.iloc[i,j]) + "Class") in dict:
                total = dict.get(df.columns[j] + str(df.iloc[i,j]) + "Class")
            else:
                total = df.value_counts([df.columns[j], "Class"])[df.iloc[i,j],1]
                dict.__setitem__(df.columns[j] + str(df.iloc[i,j]) + "Class",total)

        #Add up confidence for entry
        confidence = total/count
        entry += confidence

    #Get the average confidence for entry
    entry /= cols

    #If average confidence for entry >= 50% = malicious
    if(entry >= 0.50):
        isMalicious += 1

    #False positive (Detected malicious but is benign)
    if(entry >= 0.50 and df.iloc[i,0] == 0):
        falsePositive += 1

    #False negative (Detected benign but is malicious)
    if(entry < 0.50 and df.iloc[i,0] == 1):
        falseNegative += 1

accuracy = ((isMalicious - falsePositive) / malicious) * 100

#Print results
print("# of actual malicious: "+ str(malicious))
print("# detected to be malicious: " + str(isMalicious))
print("# of false positives: " + str(falsePositive))
print("# of false negatives: " + str(falseNegative))
print("Accuracy: " + str(accuracy) + "%")
