import os
import csv
import sys
import copy
import json
import logging
from nltk.parse.stanford import StanfordParser,StanfordDependencyParser
from transformers.file_utils import is_tf_available


logger = logging.getLogger(__name__)

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, imprisonment_label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.imprisonment_label = imprisonment_label
    
    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"
    

class InputFeature(object):
    def __init__(self, input_ids, attention_mask, token_type_ids, label, imprisonment_label, real_token_len):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label
        self.imprisonment_label = imprisonment_label
        self.real_token_len = real_token_len
    
    def __repr__(self):
        return str(self.to_json_string())
    
    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output
    
    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DataProcessor(object):
    def get_example_from_tensor_dict(self, tensor_dict):
        raise NotImplementedError
    def get_train_examples(self, data_dir):
        raise NotImplementedError
    def get_dev_examples(self, data_dir):
        raise NotImplementedError
    def get_labels(self, data_dir):
        raise NotImplementedError
    def tfds_map(self, example):
        if len(self.get_labels()) > 1:
            example.label = self.get_labels()[int(example.label)]
        return example

    
    @classmethod
    def _read_csv(cls, input_file, quotechar=None):
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                json_data = json.loads(line)
                content = json_data["fact"].replace("\n", "")
                tempterm= json_data["meta"]["term_of_imprisonment"]
                imprisonment_label = 0
                if tempterm["death_penalty"] == True:
                    imprisonment_label = -1
                elif tempterm["life_imprisonment"] == True:
                    imprisonment_label = -2
                else:
                    if tempterm["imprisonment"] > 10 * 12:
                        imprisonment_label = 1
                    elif tempterm["imprisonment"] > 7 * 12:
                        imprisonment_label = 2
                    elif tempterm["imprisonment"] > 5 * 12:
                        imprisonment_label = 3
                    elif tempterm["imprisonment"] > 3 * 12:
                        imprisonment_label = 4
                    elif tempterm["imprisonment"] > 2 * 12:
                        imprisonment_label = 5
                    elif tempterm["imprisonment"] > 1 * 12:
                        imprisonment_label = 6
                    elif tempterm["imprisonment"] > 9:
                        imprisonment_label = 7
                    elif tempterm["imprisonment"] > 6:
                        imprisonment_label = 8
                    elif tempterm["imprisonment"] > 0:
                        imprisonment_label = 9
                    else:
                        imprisonment_label = 10
                label = json_data["meta"]["accusation"][0].replace("[", "").replace("]", "")
                lines.append(label + "\t" + content + "\t" + str(imprisonment_label))
            return lines
    

def convert_examples_to_features(examples, tokenizer, max_length=512, task=None, label_list=None, imprisonment_label_list=None, output_mode=None, pad_on_left=None, pad_token=None, pad_token_segment_id=0, mask_padding_with_zero=True):
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = processor[task]()
        if label_list is None:
            label_list = processor.get_labels()
            imprisonment_label_list = processor.get_imprisonment_label()
            logger.info("Using label list %s for task %s" % (label_list, task))
            logger.info("Using label list %s for task %s" % (imprisonment_label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))
    label_map = {label: i for i, label in enumerate(label_list)}
    imprisonment_label_map = {label: i for i, label in enumerate(imprisonment_label_list)}

    features = []
    for(ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % ex_index)
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
            add_special_tokens=True,
            max_length=max_length
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]
        real_token_len = len(input_ids)

        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        padding_length = max_length - len(input_ids)

        if pad_on_left:
            input_ids = ([pad_token]* padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1]*padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        
        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        if output_mode == "classification":
            label = label_map[example.label]
            imprisonment_label = imprisonment_label_map[example.imprisonment_label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("real_token_len: %s" % (real_token_len))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))
            logger.info("imprisonment label: %s (id = %d)" % (example.imprisonment_label, imprisonment_label))
        
        features.append(
            InputFeature(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label, imprisonment_label=imprisonment_label, real_token_len=real_token_len)
        )
    
    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield({
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids
                    }, ex.label)
        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
              'attention_mask': tf.int32,
              'token_type_ids': tf.int32},
             tf.int64),
            ({'input_ids': tf.TensorShape([None]),
              'attention_mask': tf.TensorShape([None]),
              'token_type_ids': tf.TensorShape([None])},
             tf.TensorShape([])))
    return features

class THUNewsProcessor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(tensor_dict['idx'].numpy(), tensor_dict['sentence'].numpy().decode("utf-8"), None, str(tensor_dict["label"].numpy()), str(tensor_dict["imprisonment_label"].numpy()))
    
    def get_train_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "data_train.json")), "train")
    
    def get_dev_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "data_test.json")), "dev")
    
    def get_test_examples(self, data_dir):
        return self._create_examples(self._read_csv(os.path.join(data_dir, "data_test.json")), "test")
    
    def get_labels(self):
        categories = []
        with open("./dataset/CAIL2018/accu.txt", encoding="utf-8") as f:
            for line in f:
                categories.append(line.strip().split('\t')[0])
        return categories
    
    def get_imprisonment_labels(self):
        categories = []
        with open("./dataset/CAIL2018/term.txt", encoding="utf-8") as f:
            for line in f:
                categories.append(line.strip().split('\t')[0])
        return categories
        
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)

            label, content, imprisonment_label = line.strip().split('\t')
            text_a = content
            label = label
            imprisonment_label = imprisonment_label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label, imprisonment_label=imprisonment_label))
        return examples

task_num_labels = {
    "thunews": 202
}


processors = {
    "thunews": THUNewsProcessor
}

output_modes = {
    "thunews": "classification"
}