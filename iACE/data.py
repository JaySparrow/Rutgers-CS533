import os
import torch
from torch.utils.data import TensorDataset
from transformers import (
    glue_processors as processors,
    glue_output_modes as output_modes,
    glue_convert_examples_to_features as convert_examples_to_features,
    AutoTokenizer,
)

from icecream import ic

data_root_path = "../data/nlu/glue"
feature_root_path = "../data/feature/glue"

task_to_data_path = {
    'sst-2': os.path.join(data_root_path, 'SST-2')
}

def generate_tokenized_data(tokenizer, task='sst-2', max_seq_len=126, is_eval=False):
    processor = processors[task]()
    output_mode = output_modes[task]

    # read text data
    data_dir = task_to_data_path[task]
    data = (
        processor.get_dev_examples(data_dir) if is_eval else processor.get_train_examples(data_dir)
    )

    # tokenize data
    label_list = processor.get_labels()
    tokenized_data = convert_examples_to_features(
        data,
        tokenizer,
        label_list=label_list,
        max_length=max_seq_len,
        output_mode=output_mode,
    )

    return tokenized_data

def load_data(tokenizer, task='sst-2', max_seq_len=126, is_eval=False, percentage=100):
    # processor = processors[task]()
    output_mode = output_modes[task]

    cached_data_file = os.path.join(
        feature_root_path,
        "cached_{}_{}_{}_{}".format(
            "dev" if is_eval else "train",
            tokenizer.name_or_path,
            max_seq_len,
            task,
        ),
    )
    # load cached data if exists
    if os.path.exists(cached_data_file):
        print(f"loading cached data from '{cached_data_file}'...")
        features = torch.load(cached_data_file)
    # read and cache data
    else:
        print("creating data...")
        features = generate_tokenized_data(tokenizer, task, max_seq_len, is_eval)
        # save 
        torch.save(features, cached_data_file)
        print("data cached to", cached_data_file)

    # convert to Tensors
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)
    
    if task == "sst-2":
        task_name = "sst"
    else:
        raise NotImplementedError

    # load clip features
    img_features = torch.load(os.path.join(feature_root_path, task_name + ("_dev" if is_eval else "_train") + ".pt"))
    text_features = torch.load(os.path.join(feature_root_path, task_name + ("_textual_dev" if is_eval else "_textual_train") + ".pt"))

    # sample data according to percentage
    max_data_end = min(len(all_labels), len(img_features[0]), len(text_features[0]))
    sampled_data_end = min(int(len(all_labels)*percentage/100) if not is_eval else len(all_labels), max_data_end)
    dataset = TensorDataset(all_input_ids[:sampled_data_end], all_attention_mask[:sampled_data_end], all_token_type_ids[:sampled_data_end], all_labels[:sampled_data_end], img_features[0][:sampled_data_end], text_features[0][:sampled_data_end])

    return dataset

if __name__ == '__main__':
    tokenizer_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False,
        do_lower_case=True,
        cache_dir=None,
    )
    dataset = load_data(tokenizer, task='sst-2', max_seq_len=126, is_eval=True, percentage=100)
    ic(dataset)
    ic(len(dataset))
