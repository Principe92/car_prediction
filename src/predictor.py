
#CompCars Data Matcher
#This script is designed to take the output weights of Resnet18, do a PCA, then create matched and mismatched
#pairs for training and testing purposes.
#Written by Peter Rhodes and Princewill Okorie

#Import tools
import torch
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from src.reader import TrainSet, TestSet
from itertools import combinations

class CarPredictor():
    """

    """

    def __init__(self, device: str = None) -> None:
        self.device = device

    def get_train_df(self, dataset: TrainSet) -> pd.DataFrame:
        """
            Converts the train data loader into a pandas dataframe after applying PCA
        """

        vectors = []
        labels = []
        imagePaths = []

        #Retrieving weight vectors, image and label from the data
        for i in range(len(dataset)):
            vector, image, label, path = dataset[i]

            vectors.append(vector)
            labels.append(label)
            imagePaths.append(path)

        df = pd.DataFrame(torch.cat(vectors, dim=0).numpy())

        # Performing Dimensonality Reduction with PCA
        pca = PCA(n_components=100)
        data = pd.DataFrame(data=pca.fit_transform(df))
        
        #Adding the labels and path to the data that went through the PCA
        data['label'] = pd.DataFrame(data=labels)
        data['path'] = pd.DataFrame(data=imagePaths)

        #Making a copy of the dataframe to be used to create mismatched pairs latere
        data2 = data.copy()

        #Creating Matched Pairs
        match_df = pd.DataFrame()
        count = 0
        stop = 0
        while stop == 0:
            
            #Getting the label of the first row and then determining all possible unique pairs
            label = data['label'][count]
            matched_index_list = data.index[data['label'] == label].tolist()
            possible_combos = list(combinations(matched_index_list, 2))

            if len(possible_combos) != 0:
                print(f'Label: {label} | Possible combos: {len(possible_combos)}')

                #Looping through all possible unique pairs and adding them to the matched dataset
                for i in range(0, len(possible_combos)):

                    # Getting values at index (left side of pair)
                    leftside = data.iloc[possible_combos[i][0]]

                    # Getting values at index (right side of pair)
                    rightside = data.iloc[possible_combos[i][1]]

                    matched_pair = pd.concat([leftside, rightside], axis=0, ignore_index=True)
                    match_df = pd.concat([match_df, matched_pair], axis=1, ignore_index=True)

                    # print(i)

                #Deleting the rows that had the given label to reduce the search size of the dataframe
                list_to_delete = set([i[0] for i in possible_combos])

                # Getting rid of all rows with those labels
                data.drop(list_to_delete, inplace=True)
                data.reset_index(inplace=True, drop=True)  # resetting index

            else:
                # Getting rid of all rows with those labels
                data.drop(matched_index_list, inplace=True)
                data.reset_index(inplace=True, drop=True)  # resetting index

            print(f'Label: {label} | Data size: {len(data)}')

            if len(data) == 0:
                stop = 1

        # Transposing Mixed_Data into the right orientation
        match_df = match_df.T
        match_df['Match'] = 1
        match_df = match_df.rename(columns={100: 'label1', 101:'path1', 202: 'label2', 203:'path2', 204: 'Match'})




        # Randomly mixing data for the unmatched pairs
        num_non_match = len(match_df)
        non_match_df = pd.DataFrame()
        print(f'# of matched data: {num_non_match}')

        #Randomly selecting two rows and putting them together to create a mismatched pair
        while len(non_match_df.columns) < num_non_match:
            index1 = np.random.randint(0, len(data2))
            index2 = np.random.randint(0, len(data2))

            # Getting values at index (left side of pair)
            leftside = data2.iloc[index1]

            # Getting values at index (right side of pair)
            rightside = data2.iloc[index2]

            mismatched_pair = pd.concat([leftside, rightside], axis=0, ignore_index=True)
            non_match_df = pd.concat([non_match_df, mismatched_pair], axis=1, ignore_index=True)

            print(f'# Unmatched size: {len(non_match_df.columns)}')

        # Transposing non_match_df into the right orientation
        non_match_df = non_match_df.T
        non_match_df['Match'] = 0
        non_match_df = non_match_df.rename(columns={100: 'label1', 101:'path1', 202: 'label2', 203:'path2', 204: 'Match'})

        # dropping rows where matches accidently exist
        non_match_df = non_match_df[non_match_df['label1'] != non_match_df['label2']]

        # Combining the two data sets
        full_data = pd.concat([non_match_df, match_df]).sample(frac=1)
        full_data['Path'] = pd.DataFrame(full_data['path1'].astype(str) + '||' + full_data['path2'].astype(str))
        full_data.reset_index(inplace=True, drop=True)

        return full_data

    def get_test_df(self, dataset: TestSet) -> pd.DataFrame:
        """
            Converts the test data loader into a pandas dataframe after applying PCA
        """

        vectors1 = []
        vectors2 = []
        labels = []
        imagePaths = []

        #Retrieving weight vectors, image and label from the data
        for i in range(len(dataset)):
            v1, v2, _, _, y, path = dataset[i]

            vectors1.append(v1)
            vectors2.append(v2)
            labels.append(y)
            imagePaths.append(path)

        df1 = pd.DataFrame(torch.cat(vectors1, dim=0).numpy())
        df2 = pd.DataFrame(torch.cat(vectors2, dim=0).numpy())

        # Performing Dimensonality Reduction with PCA
        pca = PCA(n_components=100)

        reducedDf1 = pd.DataFrame(data=pca.fit_transform(df1))
        reducedDf2 = pd.DataFrame(data=pca.fit_transform(df2))

        # Combining the data into a full set again, shuffling it and reset index
        full_data = pd.concat([reducedDf1, reducedDf2], axis=1, join="inner")
        full_data['Match'] = pd.DataFrame(np.array(labels))
        full_data['Path'] = pd.DataFrame(np.array(imagePaths))        
        full_data.reset_index(inplace=True, drop=True)

        return full_data
