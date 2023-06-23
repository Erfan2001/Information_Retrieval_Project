import numpy as np
import pandas as pd
import os
import shutil
import json
from src.tools.logger import *


def directories_json(
    input_path: r"F:\University\Term8\Information Retrieval\Project\Dataset",
    output_path: r"F:\University\Term8\Information Retrieval\Project\src\data",
):
    allSubjects = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # if not os.path.exists(os.path.join(output_path, "train")):
    #     os.makedirs(os.path.join(output_path, "train"))
    # if not os.path.exists(os.path.join(output_path, "test")):
    #     os.makedirs(os.path.join(output_path, "test"))

    for subject in os.listdir(input_path):
        allSubjects.append(subject)

    globalCounter = 0
    document_counter = 0
    for index, subject in enumerate(allSubjects):
        logger.info("[INFO] Start saving %s document" % subject)
        # train_director = os.path.join(output_path, "train")
        train_director = output_path
        if not os.path.exists(train_director):
            os.makedirs(train_director)
        document_counter = 0

        for document in os.listdir(os.path.join(input_path, subject)):
            text = ""
            try:
                with open(
                    os.path.join(input_path, subject, document), encoding="utf-8"
                ) as f:
                    text = f.read()
            except:
                logger.error(
                    "[Error] When opening %s document in %s category"
                    % (document, subject)
                )

            # todo ! Add processed text to our cleaned dataset

            json_file = {"text": text, "label": index}

            try:
                with open(
                    os.path.join(train_director, "%d.json" % globalCounter),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(json_file, ensure_ascii=False, indent=4))
                    globalCounter += 1
                    document_counter += 1
            except:
                logger.error(
                    "[Error] When saving %s document in %s category"
                    % (document, subject)
                )

        logger.info(
            "[INFO] Finish saving %s document with %s samples"
            % (subject, document_counter)
        )
