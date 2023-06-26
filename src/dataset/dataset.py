import os
import shutil
from sklearn.model_selection import train_test_split
import uuid

# --------
from src.dataset.convertor import directories_json
from src.tools.logger import *
from src.tools.logger import *
from src.dataset.preprocessing import normalizer,tokenizer

class Dataset:
    def __init__(self, configs,tokenized_dataset_path):
        self.train_set = tokenized_dataset
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

    def split_train_dev_test(self, train_per=0.75, valid_per=0.15, test_per=0.1):
        """

        :param train_per: percentage of how many of samples should be in train set
        :param valid_per: percentage of how many of samples should be in valid set
        :param test_per: percentage of how many of samples should be in test set
        :return:
        """
        if not (
            os.path.exists(os.path.join(self.processed_data, "train"))
            and os.path.exists(os.path.join(self.processed_data, "test"))
            and os.path.exists(os.path.join(self.processed_data, "valid"))
        ):
            if not os.path.exists(os.path.join(self.processed_data, "train")):
                os.makedirs(os.path.join(self.processed_data, "train"))
            if not os.path.exists(os.path.join(self.processed_data, "valid")):
                os.makedirs(os.path.join(self.processed_data, "valid"))
            if not os.path.exists(os.path.join(self.processed_data, "test")):
                os.makedirs(os.path.join(self.processed_data, "test"))
            allData = []
            input_path = os.path.join(self.processed_data, "out")
            for subject in os.listdir(input_path):
                logger.info("[INFO] Start dividing %s category to 3 parts (train - test - validation)" % (subject))
                for document in os.listdir(os.path.join(input_path, subject)):
                    allData.append(os.path.join(input_path, subject, document))
                X_train_eval, X_test = train_test_split(
                    allData, test_size=test_per, random_state=20
                )
                X_train, X_valid = train_test_split(
                    X_train_eval, test_size=valid_per, random_state=20
                )
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
                logger.info("[INFO] Finish dividing %s category to 3 parts (train - test - validation)" % (subject))
                allData.clear()
                for train_document in os.listdir(os.path.join(self.processed_data, "train")):
                    src = os.path.join(os.path.join(self.processed_data, "train"), train_document)
                    dst = os.path.join(os.path.join(self.processed_data, "train"), str(uuid.uuid1()) + ".json")
                    os.rename(src, dst)
                for validation_document in os.listdir(os.path.join(self.processed_data, "valid")):
                    src = os.path.join(os.path.join(self.processed_data, "valid"), validation_document)
                    dst = os.path.join(os.path.join(self.processed_data, "valid"),str(uuid.uuid1()) + ".json")
                    os.rename(src, dst)
                for test_document in os.listdir(os.path.join(self.processed_data, "test")):
                    src = os.path.join(os.path.join(self.processed_data, "test"), test_document)
                    dst = os.path.join(os.path.join(self.processed_data, "test"), str(uuid.uuid1()) + ".json")
                    os.rename(src, dst)
        shutil.rmtree(os.path.join(self.processed_data, "out"))
        return self


def prepare_dataset(configs):
    # Normalizer
    normalizer(os.path.join(configs.data_dir, "train"),os.path.join(configs.cache_dir,'normalized_dataset'))
    # Tokenizer
    tokenizer(os.path.join(configs.cache_dir,'normalized_dataset'),os.path.join(configs.cache_dir,'tokenized_dataset'))
    # Convert to json
    dataset = Dataset(configs,os.path.join(configs.cache_dir,'tokenized_dataset')).load_dataset_from_file()
    # Split data to 3 parts
    dataset.split_train_dev_test()
    return
