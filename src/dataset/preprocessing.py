from hazm import *
import os

def normalizer(input_path,output_path):
    """
        Normalize dataset by using Hazm Library
    """
    normalizer = Normalizer()

    for subDir in os.listdir(input_path):
        #create root
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        #create directories in root
        if not os.path.exists(os.path.join(output_path, f'{subDir}')):
            os.makedirs(os.path.join(output_path, f'{subDir}'))
        
        #normalize files of each directory
        for root, dirs, files in os.walk(os.path.join(input_path, subDir)):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    with open(path, 'r') as f:
                        text = f.read()
                        normText = normalizer.normalize(text)
                    
                    path = os.path.join(output_path, subDir, file)
                    with open(path, 'w') as f:
                        f.write(normText)


def tokenizer(input_path,output_path):
    """
        Tokenize dataset by using Hazm Library
    """
    for subDir in os.listdir(input_path):
        #create root
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        #create directories in root
        if not os.path.exists(os.path.join(output_path, f'{subDir}')):
            os.makedirs(os.path.join(output_path, f'{subDir}'))
        
        #tokenize files of each directory
        for root, dirs, files in os.walk(os.path.join(input_path, subDir)):
            for file in files:
                if file.endswith(".txt"):
                    path = os.path.join(root, file)
                    with open(path, 'r') as f:
                        text = f.read()
                        tokenText = word_tokenize(text)
                    
                    path = os.path.join(output_path, subDir, file)
                    with open(path, 'w') as f:
                        f.write(str(tokenText))
