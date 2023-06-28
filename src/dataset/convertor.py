import numpy as np
import pandas as pd
import os
import shutil
import json
from src.tools.logger import *
from src.dataset.dataset import purifying_text,remove_phrases

def directories_json(input_path, output_path):
    """
        Change all subject folders,containing lots of documents to json formt
        Each json has two key/value:
            1) test: preprocessed text
            2) label: integer label due to label2id function
    """
    allSubjects = []
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for subject in os.listdir(input_path):
        allSubjects.append(subject)

    document_counter = 0
    for index, subject in enumerate(allSubjects):
        logger.info("[INFO] Start saving %s document" % subject)
        if not os.path.exists(os.path.join(output_path, subject)):
            os.makedirs(os.path.join(output_path, subject))
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
            text=remove_phrases(text)
            text = purifying_text(cleaning)
            json_file = {"text": text, "label": index}

            try:
                with open(
                    os.path.join(
                        os.path.join(output_path, subject), "%d.json" % document_counter
                    ),
                    "w",
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(json_file, ensure_ascii=False, indent=4))
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