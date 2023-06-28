import os
import shutil
from sklearn.model_selection import train_test_split
import uuid
from datasets import load_dataset
from cleantext import clean

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


def remove_phrases(text):
    # Define the phrases 
    phrases_to_remove = ['به گزارش', 'به نقل از', 'به عنوان مثال', 'به طور کلی']
    # Create a regular expression pattern from the phrases to remove
    pattern = '|'.join(phrases_to_remove)
    # Use the re.sub() function to replace the pattern with an empty string
    text = re.sub(pattern, '', text)
    return text
    
def purifying_text(text):
    text = text.strip()

    text = clean(text,
        no_digits=False,
        fix_unicode=True,
        lower=True,
        to_ascii=False,
        no_line_breaks=True,
        no_emails=True,
        no_urls=True,
        no_numbers=False,
        no_phone_numbers=True,
        no_currency_symbols=True,
        replace_with_number="",
        replace_with_url="",
        no_punct=False,
        replace_with_digit="0",
        replace_with_email="",
    )

    html_Cleaner = re.compile('<.*?>')
    text = re.sub(html_Cleaner, '', text)

    # removing some specific patterns (In Persian mostly)
    weird_pattern = re.compile("["
        u"\u2640-\u2642"
        u"\u2600-\u2B55"
        u"\U0001F600-\U0001F64F" 
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u'\U00010000-\U0010ffff'
        u"\u200d"
        u"\u23cf"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\u23e9"
        u"\u231a"
        u"\u3030"
        u"\ufe0f"
        u"\u2069"
        u"\u2066"
        u"\u2068"
        u"\u2067"
        "]+", flags=re.UNICODE)
    text = weird_pattern.sub(r'', text)
    # Remove spaces
    text = re.sub("#", "", text)
    text = re.sub("\s+", " ", text)
    return text

def prepare_dataset(configs):
    """
        Prepare dataset before training (All preprocessed sections)
    """
    # Normalizer
    normalizer(os.path.join(configs.data_dir, "train"),os.path.join(configs.cache_dir,'normalized_dataset'))
    # Tokenizer
    tokenizer(os.path.join(configs.cache_dir,'normalized_dataset'),os.path.join(configs.cache_dir,'tokenized_dataset'))
    # Convert to json
    dataset = Dataset(configs,os.path.join(configs.cache_dir,'tokenized_dataset')).load_dataset_from_file()
    # Split data to 3 parts
    dataset.split_train_dev_test()
    return

def load_dataset_HF(input_path):
    """
        Change dataset to the HF format
    """
    loaded_dataset= load_dataset(input_path)
    return loaded_dataset

def label2id(input_path):
    """
        Make dictionary consists of key which is labels and value which is ids
    """
    dic={}
    count=0
    for subject in os.walk(input_path):
        dic[subject]=count
        count+=1
    return dic

def id2label(input_path):
    """
        Make dictionary consists of key which is id and value which is label
    """
    dic={}
    count=0
    for subject in os.walk(input_path):
        dic[count]=subject
        count+=1
    return dic
