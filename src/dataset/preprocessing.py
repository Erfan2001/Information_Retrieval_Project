from hazm import *
import os


def preprocessor():
    
    input = "/Users/aminh/Desktop/IR-project/dataset/"
    output = "/Users/aminh/Desktop/IR-project/Information_Retrieval_Project/NormalizeDataset/"
    normalizer = Normalizer()

    for subDir in os.listdir(input):
        #create root
        if not os.path.exists(output):
            os.makedirs(output)
        
        #create directories in root
        if not os.path.exists(os.path.join(output, f'{subDir}')):
            os.makedirs(os.path.join(output, f'{subDir}'))
        
        #normalize files of each directory
        for root, dirs, files in os.walk(os.path.join(input, subDir)):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    with open(path, 'r') as f:
                        text = f.read()
                        normText = normalizer.normalize(text)
                    
                    path = os.path.join(output, subDir, file)
                    with open(path, 'w') as f:
                        f.write(normText)


def tokenizer():
    input = "/Users/aminh/Desktop/IR-project/Information_Retrieval_Project/NormalizeDataset/"
    output = "/Users/aminh/Desktop/IR-project/Information_Retrieval_Project/TokenizeDataset/"

    for subDir in os.listdir(input):
        #create root
        if not os.path.exists(output):
            os.makedirs(output)
        
        #create directories in root
        if not os.path.exists(os.path.join(output, f'{subDir}')):
            os.makedirs(os.path.join(output, f'{subDir}'))
        
        #tokenize files of each directory
        for root, dirs, files in os.walk(os.path.join(input, subDir)):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    with open(path, 'r') as f:
                        text = f.read()
                        tokenText = word_tokenize(text)
                    
                    path = os.path.join(output, subDir, file)
                    with open(path, 'w') as f:
                        f.write(str(tokenText))


preprocessor()
tokenizer()