import pandas as pd

df = pd.read_csv("Labels.csv")
cols = len(df.columns)
dict = {}
malicious = 0
isMalicious = 0


for i in range(5000):
    entry = 0
    if(df.iloc[i,0] == 1):
        malicious += 1

    for j in range(1,cols):
        if (df.columns[j]+str(df.iloc[i,j])) in dict:
            count = dict.get(df.columns[j]+str(df.iloc[i,j]))
        else:
            count = df.value_counts(df.columns[j])[df.iloc[i,j]]
            dict.__setitem__(df.columns[j]+str(df.iloc[i,j]), count)

        if (df.columns[j] + str(df.iloc[i,j]) + "Class") in dict:
            total = dict.get(df.columns[j] + str(df.iloc[i,j]) + "Class")
        else:
            total = df.value_counts([df.columns[j], "Class"])[df.iloc[i,j],1]
            dict.__setitem__(df.columns[j] + str(df.iloc[i,j]) + "Class",total)
        confidence = total/count
        entry += confidence
    entry /= cols
    if(entry > 0.50):
            isMalicious += 1
    
print("# of actual malicious: "+ str(malicious))
print("# detected to be malicious: " + str(isMalicious))

