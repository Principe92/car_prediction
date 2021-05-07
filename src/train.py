
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

        for i in range(len(dataset)):
            vector, image, label = dataset[i]

            vectors.append(vector)
            labels.append(label)

        df = pd.DataFrame(torch.cat(vectors, dim=0).numpy())
        ds_size = df.shape[0]  # 4352
        ds_size_hf = int(ds_size / 4)

        pca = PCA(n_components=100)

        reducedDf1 = pd.DataFrame(data=pca.fit_transform(df))
        reducedDf1['label'] = pd.DataFrame(np.array(labels))

        mixed_data = pd.DataFrame()
        count = 0
        while count < ds_size_hf:
            index = np.random.randint(0, len(reducedDf1))

            first_row = reducedDf1.iloc[index]
            first_label = reducedDf1['label'][index]  # getting the first label

            reducedDf1.drop(index, inplace=True)

            # Dropping the original row so it isn't found multiple times
            reducedDf1.reset_index(inplace=True, drop=True)

            for row in range(0, len(reducedDf1)):
                if reducedDf1['label'][row] == first_label:
                    second_row = reducedDf1.iloc[row]

                    # Dropping the original row so it isn't found multiple times
                    reducedDf1.drop(row)

                    # Combining the two rows into one and adding the match label
                    first_row = first_row.append(second_row)
                    first_row['Match'] = 1
                    first_row = first_row.reset_index(drop=True)

                    # adding the original to the new df
                    mixed_data = pd.concat(
                        [mixed_data, first_row], axis=1, ignore_index=True)

                    count = count+1
                    break

        # Transposing Mixed_Data into the right orientation
        mixed_data = mixed_data.T
        mixed_data = mixed_data.rename(
            columns={100: 'label1', 201: 'label2', 202: 'Match'})
        columns_name = mixed_data.columns

        # Matching the other half of the data randomly (how random depends on if the data was shuffled before hand)
        df_first_half = reducedDf1[0:ds_size_hf]
        df_first_half.reset_index(inplace=True, drop=True)
        df_second_half = reducedDf1[ds_size_hf: int(ds_size/2) + 1]
        df_second_half.reset_index(inplace=True, drop=True)

        # combining the unpaired data
        mismatched_df = pd.concat(
            [df_first_half, df_second_half], axis=1, join="inner")
        mismatched_df['Match'] = 0
        mismatched_df.columns = columns_name

        # Combining the data into a full set again, shuffling it and reset index
        full_data = pd.concat([mismatched_df, mixed_data]).sample(frac=1)
        full_data.reset_index(inplace=True, drop=True)

        return full_data

    def get_test_df(self, dataset: TestSet) -> pd.DataFrame:
        """
            Converts the test data loader into a pandas dataframe after applying PCA
        """

        vectors1 = []
        vectors2 = []
        labels = []

        for i in range(len(dataset)):
            v1, v2, _, _, y = dataset[i]

            vectors1.append(v1)
            vectors2.append(v2)
            labels.append(y)

        df1 = pd.DataFrame(torch.cat(vectors1, dim=0).numpy())
        df2 = pd.DataFrame(torch.cat(vectors2, dim=0).numpy())

        pca = PCA(n_components=100)

        reducedDf1 = pd.DataFrame(data=pca.fit_transform(df1))
        reducedDf2 = pd.DataFrame(data=pca.fit_transform(df2))

        # combining the unpaired data
        mismatched_df = pd.concat(
            [reducedDf1, reducedDf2], axis=1, join="inner")
        mismatched_df['Match'] = pd.DataFrame(np.array(labels))
        # mismatched_df.columns = columns_name

        # Combining the data into a full set again, shuffling it and reset index
        full_data = mismatched_df.copy()
        full_data.reset_index(inplace=True, drop=True)

        return full_data

