import os
import shutil
import uuid

def flat_files(input_path,output_path):
    """
        Flat all the files in each category to the specific folder and put them near each other
    """
    for subject in os.listdir(input_path):
        allData=[]
        for doc in os.listdir(os.path.join(input_path,subject)):
            shutil.copy(os.path.join(input_path,subject,doc), output_path)
        for x in os.listdir(output_path):
            os.rename(os.path.join(output_path,x),os.path.join(output_path,str(uuid.uuid1()) + ".txt"))

def pytorch_dataset(labels_dict, input_path):
    """
        Making dataset which can be used for pytorch training
        Each item in this dataset is a tuple:
            1) First argument: text
            2) Second argument: label
    """
    total_list = []
    for dirname in os.listdir(input_path):
        f = os.path.join(input_path, dirname)
        for filename in os.listdir(f):
            f = os.path.join(f, filename)
            if os.path.isfile(f):
                #open text file in read mode
                text_file = open(f, "r")

                data = text_file.read()
                text_file.close()

                total_list.append((data, labels_dict[dirname]))
    return total_list