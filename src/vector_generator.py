
from numpy.core.numeric import full
import torch
from torch.nn import Module
from src.utils import get_acc
from torch.autograd import Variable
from torchvision import models
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
from src.reader import TrainSet, TestSet
import string
from itertools import combinations
from numba import jit, cuda


class CarPredictor():
    """

    """

    def __init__(self, device: str = None) -> None:
        self.device = device

    @jit()
    def get_train_ds(self, dataset: TrainSet):
        vectors = []
        labels = []
        imagePaths = []

        for i in range(len(dataset)):
            vector, image, label, path = dataset[i]

            vectors.append(vector)
            labels.append(label)
            imagePaths.append(path)

        return vectors, labels, imagePaths

    # @jit(nopython=True)
    def get_train_df(self, dataset: TrainSet) -> pd.DataFrame:
        """
            Converts the train data loader into a pandas dataframe after applying PCA
        """

        vectors, labels, imagePaths = self.get_train_ds(dataset)

        df = pd.DataFrame(torch.cat(vectors, dim=0).numpy())

        # Performing Dimensonality Reduction with PCA
        pca = PCA(n_components=100)

        # Df1 PCA
        data = pd.DataFrame(data=pca.fit_transform(df))
        data['label'] = pd.DataFrame(data=labels)
        data['path'] = pd.DataFrame(data=imagePaths)

        return self.perform_train_df(data)

    
    @jit()
    def perform_train_df(self, data: pd.DataFrame) -> pd.DataFrame:
        data2 = data.copy()

        match_df = pd.DataFrame()
        count = 0
        stop = 0
        while stop == 0:
            label = data['label'][count]
            matched_index_list = data.index[data['label'] == label].tolist()
            possible_combos = list(combinations(matched_index_list, 2))

            if len(possible_combos) != 0:
                print('Label: %f | Possible combos: %f'.format(label, (len(possible_combos))))

                for i in range(0, len(possible_combos)):

                    # Getting values at index (left side of pair)
                    leftside = data.iloc[possible_combos[i][0]]

                    # Getting values at index (right side of pair)
                    rightside = data.iloc[possible_combos[i][1]]

                    matched_pair = pd.concat([leftside, rightside], axis=0, ignore_index=True)
                    match_df = pd.concat([match_df, matched_pair], axis=1, ignore_index=True)

                    # print(i)

                list_to_delete = set([i[0] for i in possible_combos])

                # Getting rid of all rows with those labels
                data.drop(list_to_delete, inplace=True)
                data.reset_index(inplace=True, drop=True)  # resetting index

            else:
                # Getting rid of all rows with those labels
                data.drop(matched_index_list, inplace=True)
                data.reset_index(inplace=True, drop=True)  # resetting index

            print('Label: %f | Data size: %f'.format(label, (len(data))))

            if len(data) == 0:
                stop = 1

        # Transposing Mixed_Data into the right orientation
        match_df = match_df.T
        match_df['Match'] = 1
        match_df = match_df.rename(columns={100: 'label1', 101:'path1', 202: 'label2', 203:'path2', 204: 'Match'})

        # Randomly mixing data for the rest
        num_non_match = len(match_df)
        non_match_df = pd.DataFrame()
        print('# of matched data: %f'.format(num_non_match))

        while len(non_match_df.columns) < num_non_match:
            index1 = np.random.randint(0, len(data2))
            index2 = np.random.randint(0, len(data2))

            # Getting values at index (left side of pair)
            leftside = data2.iloc[index1]

            # Getting values at index (right side of pair)
            rightside = data2.iloc[index2]

            mismatched_pair = pd.concat([leftside, rightside], axis=0, ignore_index=True)
            non_match_df = pd.concat([non_match_df, mismatched_pair], axis=1, ignore_index=True)

            print('# Unmatched size: %s'.format(len(non_match_df.columns)))

        # Transposing non_match_df into the right orientation
        non_match_df = non_match_df.T
        non_match_df['Match'] = 0
        non_match_df = non_match_df.rename(columns={100: 'label1', 101:'path1', 202: 'label2', 203:'path2', 204: 'Match'})

        # drop where matches accidently exist
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

        for i in range(len(dataset)):
            v1, v2, _, _, y, path = dataset[i]

            vectors1.append(v1)
            vectors2.append(v2)
            labels.append(y)
            imagePaths.append(path)

        df1 = pd.DataFrame(torch.cat(vectors1, dim=0).numpy())
        df2 = pd.DataFrame(torch.cat(vectors2, dim=0).numpy())

        pca = PCA(n_components=100)

        reducedDf1 = pd.DataFrame(data=pca.fit_transform(df1))
        reducedDf2 = pd.DataFrame(data=pca.fit_transform(df2))

        # Combining the data into a full set again, shuffling it and reset index
        full_data = pd.concat([reducedDf1, reducedDf2], axis=1, join="inner")
        full_data['Match'] = pd.DataFrame(np.array(labels))
        full_data['Path'] = pd.DataFrame(np.array(imagePaths))        
        full_data.reset_index(inplace=True, drop=True)

        return full_data
