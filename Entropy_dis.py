from numpy.core.defchararray import count
import pandas as pd
import numpy as np
import numpy as np
from math import log2


def main():
    s = pd.read_csv('/Users/manish/Desktop/workspace/data_mining/A1-dm.csv')
    print("******************************************************")
    print("Entropy Discretization                         STARTED")
    s = entropy_discretization(s)
    print("Entropy Discretization                         COMPLETED")


# This method discretizes attribute A1
# If the information gain is 0, i.e the number of
# distinct class is 1 or
# If min f/ max f < 0.5 and the number of distinct values is floor(n/2)
# Then that partition stops splitting.
# This method discretizes s A1
# If the information gain is 0, i.e the number of
# distinct class is 1 or
# If min f/ max f < 0.5 and the number of distinct values is floor(n/2)
# Then that partition stops splitting.
def entropy_discretization(s):
    I = {}
    i = 0
    while (uniqueValue(s)):
        # Step 1: pick a threshold
        threshold = s['A1'].iloc[0]

        # Step 2: Partititon the data set into two parttitions
        s1 = s[s['A1'] < threshold]
        print("s1 after spitting")
        print(s1)
        print("******************")
        s2 = s[s['A1'] >= threshold]
        print("s2 after spitting")
        print(s2)
        print("******************")

        # Step 3: calculate the information gain.
        informationGain = information_gain(s1, s2, s)
        I.update({f'informationGain_{i}': informationGain, f'threshold_{i}': threshold})
        print(f'added informationGain_{i}: {informationGain}, threshold_{i}: {threshold}')
        s = s[s['A1'] != threshold]
        i += 1

    # Step 5: calculate the min information gain
    n = int(((len(I) / 2) - 1))
    print("Calculating minimum threshold")
    print("*****************************")
    minInformationGain = 0
    minThreshold = 0
    for i in range(0, n):
        if (I[f'informationGain_{i}'] < minInformationGain):
            minInformationGain = I[f'informationGain_{i}']
            minThreshold = I[f'threshold_{i}']

    print(f'minThreshold: {minThreshold}, minInformationGain: {minInformationGain}')

    # Step 6: keep the partitions of S based on the value of threshold_i
    minPartition(minInformationGain, minThreshold, s, s1, s2)


def uniqueValue(s):
    # are records in s the same? return true
    if s.nunique()['A1'] == 1:
        return False
    # otherwise false
    else:
        return True


def minPartition(minInformationGain, minThreshold, s, s1, s2):
    print(f'informationGain: {minInformationGain}, threshold: {minThreshold}')
    merged_partitions = pd.merge(s1, s2)
    merged_partitions = pd.merge(merged_partitions, s)
    print("Best Partition")
    print("***************")
    print(merged_partitions)
    print("***************")
    return merged_partitions


def information_gain(s1, s2, s):
    # calculate cardinality for s1
    cardinalityS1 = len(pd.Index(s1['A1']).value_counts())
    print(f'The Cardinality of s1 is: {cardinalityS1}')
    # calculate cardinality for s2
    cardinalityS2 = len(pd.Index(s2['A1']).value_counts())
    print(f'The Cardinality of s2 is: {cardinalityS2}')
    # calculate cardinality of s
    cardinalityS = len(pd.Index(s['A1']).value_counts())
    print(f'The Cardinality of s is: {cardinalityS}')
    # calculate informationGain
    informationGain = (cardinalityS1 / cardinalityS) * entropy(s1) + (cardinalityS2 / cardinalityS) * entropy(s2)
    print(f'The total informationGain is: {informationGain}')
    return informationGain


def entropy(s):
    print("calculating the entropy for s")
    print("*****************************")
    print(s)
    print("*****************************")

    # initialize ent
    ent = 0

    # calculate the number of classes in s
    numberOfClasses = s['Class'].nunique()
    print(f'Number of classes for dataset: {numberOfClasses}')
    value_counts = s['Class'].value_counts()
    p = []
    for i in range(0, numberOfClasses):
        n = s['Class'].count()
        # calculate the frequency of class_i in S1
        print(f'p{i} {value_counts.iloc[i]}/{n}')
        f = value_counts.iloc[i]
        pi = f / n
        p.append(pi)

    print(p)

    for pi in p:
        ent += -pi * log2(pi)

    return ent


main()