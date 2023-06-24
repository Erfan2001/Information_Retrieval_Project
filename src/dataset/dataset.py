import os
import shutil
from sklearn.model_selection import train_test_split

# --------
from src.dataset.convertor import directories_json
from src.tools.logger import *


class Dataset:
    def __init__(self, configs):
        self.train_set = os.path.join(configs.data_dir, "train")
        self.processed_data = configs.cache_dir
        self.test_set = os.path.join(configs.data_dir, "test")

    def load_dataset_from_file(self):
        """
        :param file_path: data_dir
        :return:
        """
        if not (os.path.exists(os.path.join(self.processed_data, "out"))):
            os.makedirs(os.path.join(self.processed_data, "out"))
            logger.info(
                "[INFO] Start changing directories to json from %s path ",
                self.train_set,
            )
            directories_json(self.train_set, os.path.join(self.processed_data, "out"))
            logger.info(
                "[INFO] Finish changing directories to json from %s path ",
                self.train_set,
            )

        return self

    def split_train_dev_test(self, train_per=0.7, valid_per=0.15, test_per=0.15):
        """

        :param train_per: percentage of how many of samples should be in train set
        :param valid: percentage of how many of samples should be in valid set
        :param test: percentage of how many of samples should be in test set
        :return:
        """
        if not (
            os.path.exists(os.path.join(self.processed_data, "train"))
            and os.path.exists(os.path.join(self.processed_data, "test"))
            and os.path.exists(os.path.join(self.processed_data, "valid"))
        ):
            allData = []
            train_path = os.path.join(self.processed_data, "out")
            for item in os.listdir(train_path):
                allData.append(os.path.join(train_path, item))
            X_train_eval, X_test = train_test_split(
                allData, test_size=test_per, random_state=20
            )
            X_train, X_valid = train_test_split(
                X_train_eval, test_size=valid_per, random_state=20
            )
            if not os.path.exists(os.path.join(self.processed_data, "train")):
                os.makedirs(os.path.join(self.processed_data, "train"))
            if not os.path.exists(os.path.join(self.processed_data, "valid")):
                os.makedirs(os.path.join(self.processed_data, "valid"))
            if not os.path.exists(os.path.join(self.processed_data, "test")):
                os.makedirs(os.path.join(self.processed_data, "test"))

            for item in allData:
                if item in X_train:
                    shutil.move(
                        item,
                        os.path.join(self.processed_data, "train"),
                    )
                elif item in X_valid:
                    shutil.move(
                        item,
                        os.path.join(self.processed_data, "valid"),
                    )
                elif item in X_test:
                    shutil.move(
                        item,
                        os.path.join(self.processed_data, "test"),
                    )
        return self


def prepare_dataset(configs):
    dataset = Dataset(configs).load_dataset_from_file()
    dataset.split_train_dev_test()
    return 
