import csv
import glob
import json
import logging
import os
from typing import List
import numpy as np

import tqdm
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None, ques_types="", extend_context=""):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label
        self.ques_types = ques_types
        self.extend_context = extend_context


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {"input_ids": input_ids, "input_mask": input_mask, "segment_ids": segment_ids}
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class RaceProcessor(DataProcessor):
    """Processor for the race-data data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        high = os.path.join(data_dir, "train/high")
        middle = os.path.join(data_dir, "train/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        high = os.path.join(data_dir, "dev/high")
        middle = os.path.join(data_dir, "dev/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        high = os.path.join(data_dir, "test/high")
        middle = os.path.join(data_dir, "test/middle")
        high = self._read_txt(high)
        middle = self._read_txt(middle)
        return self._create_examples(high + middle, "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]

    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm.tqdm(files, desc="read files"):
            with open(file, "r", encoding="utf-8") as fin:
                data_raw = json.load(fin)
                data_raw["race_id"] = file
                lines.append(data_raw)
        return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            race_id = "%s-%s" % (set_type, data_raw["race_id"])
            article = data_raw["article"]
            for i in range(len(data_raw["answers"])):
                truth = str(ord(data_raw["answers"][i]) - ord("A"))
                question = data_raw["questions"][i]
                options = data_raw["options"][i]

                examples.append(
                    InputExample(
                        example_id=race_id,
                        question=question,
                        contexts=[article, article, article, article],  # this is not efficient but convenient
                        endings=[options[0], options[1], options[2], options[3]],
                        label=truth,
                    )
                )
        return examples


class ReclorProcessor(DataProcessor):
    """Processor for the ReClor data set."""

    def get_train_examples(self, data_dir, version=1):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        ques_types = np.load(os.path.join(data_dir, "train_ques_types.npy"))
        print("load context cp data")
        extend_context = np.load(os.path.join(data_dir, "train_extended_context_cp_v"+ str(version) +".npy"), allow_pickle=True)
        return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train", ques_types, extend_context)

    def get_dev_examples(self, data_dir, version=1):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        ques_types = np.load(os.path.join(data_dir, "val_ques_types.npy"))
        extend_context = np.load(os.path.join(data_dir, "val_extended_context_cp_v"+ str(version) +".npy"), allow_pickle=True)
        return self._create_examples(self._read_json(os.path.join(data_dir, "val.json")), "dev", ques_types, extend_context)

    def get_test_examples(self, data_dir, version=1):
        logger.info("LOOKING AT {} test".format(data_dir))
        ques_types = np.load(os.path.join(data_dir, "test_ques_types.npy"))
        extend_context = np.load(os.path.join(data_dir, "test_extended_context_cp_v"+ str(version) +".npy"), allow_pickle=True)
        return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test", ques_types, extend_context)

    def get_labels(self):
        """See base class."""
        return [0, 1, 2, 3]

    def _read_json(self, input_file):
        with open(input_file, "r") as f:
            lines = json.load(f)
        return lines

    def _create_examples(self, lines, type, question_types, extend_contexts):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, d in enumerate(lines):
            context = d['context']
            question = d['question']
            answers = d['answers']
            label = 0 if type == "test" else d['label'] # for test set, there is no label. Just use 0 for convenience.
            # label = d['label'] # for test set, there is no label. Just use 0 for convenience.
            id_string = d['id_string']
            examples.append(
                InputExample(
                    example_id = id_string,
                    question = question,
                    contexts=[context, context, context, context],  # this is not efficient but convenient
                    endings=[answers[0], answers[1], answers[2], answers[3]],
                    label = label,
                    ques_types= question_types[i],
                    extend_context = [extend_contexts[i][0], extend_contexts[i][1], extend_contexts[i][2], extend_contexts[i][3]]
                    )
                )  
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
    ques_type_before=1,
    whether_extend_context=False,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    all_length = list()
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                # this is for cloze question
                if ques_type_before:
                    # text_b = example.ques_types + " " + example.question.replace("_", ending)
                    text_b = example.question.replace("_", ending)
                else:
                    # text_b = example.question.replace("_", ending) + " " + example.ques_types
                    text_b = example.question.replace("_", ending)
            else:
                if ques_type_before:
                    # text_b = example.ques_types + " " + example.question + " " + ending
                    text_b = example.question + " " + ending
                else:
                    # text_b = example.question + " " + example.ques_types + " " + ending
                    text_b = example.question + " " + ending

            if whether_extend_context:
                text_b = text_b + " " + tokenizer.additional_special_tokens[0] + " " + example.extend_context[ending_idx]

            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True, max_length=max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=True)
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and race-data and you are poping question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            input_ids, attention_mask = inputs["input_ids"], inputs['attention_mask']
            token_type_ids = inputs["token_type_ids"]

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length

            choices_features.append((input_ids, attention_mask, token_type_ids))

        label = label_map[example.label]

        features.append(InputFeatures(example_id=example.example_id, choices_features=choices_features, label=label,))

    return features


processors = {"race": RaceProcessor, "reclor": ReclorProcessor}
MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"race": 4, "reclor": 4}
