from PIL import Image
from icecream import ic
import clip
import re
import os
import csv
import torch
import argparse
torch.set_num_threads(4)
clip_model, preprocess = clip.load('ViT-B/32', "cuda:0")
vq_parser = argparse.ArgumentParser(description='extract visual textual features')
vq_parser.add_argument('-task', default="sst")
vq_parser.add_argument("-eval",  "--eval", action='store_true', help="eval", dest='eval')
vq_parser.add_argument("-nlu_dataset",  "--nlu_dataset", type=str, default="glue")
# vq_parser.add_argument("-extract_visual",  "--extract_visual", action='store_true', help="eval", dest='eval')
# vq_parser.add_argument("-extract_textual",  "--extract_textual", action='store_true', help="eval", dest='eval')
args = vq_parser.parse_args()
visual_data_root_path = "../data/ima_gan/{}/output/".format(args.nlu_dataset)
textual_data_root_path = "../data/nlu/{}/".format(args.nlu_dataset)
save_root_path = "../data/feature/{}/".format(args.nlu_dataset)
if not os.path.exists(save_root_path): os.makedirs(save_root_path)

task_to_path = {
    "cola": visual_data_root_path + "glue_cola",
    "mnli": visual_data_root_path + "glue_mnli",
    "mnli_m": visual_data_root_path + "glue_mnli_m",
    "mnli_mm": visual_data_root_path + "glue_mnli_mm",
    "mrpc": visual_data_root_path + "glue_mrpc",
    "qnli": visual_data_root_path + "glue_qnli",
    "qqp": visual_data_root_path + "glue_qqp",
    "rte": visual_data_root_path + "glue_rte",
    "sst": visual_data_root_path + "glue_sst",
    "sts": visual_data_root_path + "glue_sts",
    "wnli": visual_data_root_path + "glue_wnli",
    "swag": visual_data_root_path + "swag",
    "squad1": visual_data_root_path + "squad1",
    "squad2": visual_data_root_path + "squad2",
}

registered_path = {
    'glue_cola_train': os.path.join(textual_data_root_path, "CoLA/train.tsv"),
    'glue_cola_dev': os.path.join(textual_data_root_path, "CoLA/dev.tsv"),
    'glue_cola_test': os.path.join(textual_data_root_path, "CoLA/test.tsv"),

    'glue_mnli_train': os.path.join(textual_data_root_path, "MNLI/train.tsv"),
    'glue_mnli_m_dev': os.path.join(textual_data_root_path, "MNLI/dev_matched.tsv"),
    'glue_mnli_mm_dev': os.path.join(textual_data_root_path, "MNLI/dev_mismatched.tsv"),
    'glue_mnli_m_test': os.path.join(textual_data_root_path, "MNLI/test_matched.tsv"),
    'glue_mnli_mm_test': os.path.join(textual_data_root_path, "MNLI/test_mismatched.tsv"),

    'glue_mrpc_train': os.path.join(textual_data_root_path, "MRPC/train.tsv"),
    'glue_mrpc_dev': os.path.join(textual_data_root_path, "MRPC/dev.tsv"),
    'glue_mrpc_test': os.path.join(textual_data_root_path, "MRPC/test.tsv"),

    'glue_sst_train': os.path.join(textual_data_root_path, "SST-2/train.tsv"),
    'glue_sst_dev': os.path.join(textual_data_root_path, "SST-2/dev.tsv"),
    'glue_sst_test': os.path.join(textual_data_root_path, "SST-2/test.tsv"),

    'glue_sts_train': os.path.join(textual_data_root_path, "STS-B/train.tsv"),
    'glue_sts_dev': os.path.join(textual_data_root_path, "STS-B/dev.tsv"),
    'glue_sts_test': os.path.join(textual_data_root_path, "STS-B/test.tsv"),

    'glue_rte_train': os.path.join(textual_data_root_path, "RTE/train.tsv"),
    'glue_rte_dev': os.path.join(textual_data_root_path, "RTE/dev.tsv"),
    'glue_rte_test': os.path.join(textual_data_root_path, "RTE/test.tsv"),

    'glue_wnli_train': os.path.join(textual_data_root_path, "WNLI/train.tsv"),
    'glue_wnli_dev': os.path.join(textual_data_root_path, "WNLI/dev.tsv"),
    'glue_wnli_test': os.path.join(textual_data_root_path, "WNLI/test.tsv"),

    'glue_qnli_train': os.path.join(textual_data_root_path, "QNLI/train.tsv"),
    'glue_qnli_dev': os.path.join(textual_data_root_path, "QNLI/dev.tsv"),
    'glue_qnli_test': os.path.join(textual_data_root_path, "QNLI/test.tsv"),

    'glue_qqp_train': os.path.join(textual_data_root_path, "QQP/train.tsv"),
    'glue_qqp_dev': os.path.join(textual_data_root_path, "QQP/dev.tsv"),
    'glue_qqp_test': os.path.join(textual_data_root_path, "QQP/test.tsv"),
}

task_to_number = {
    "mnli":392703,
    "qnli":104744,
    "qqp":363847,
    "swag":73547,
    "squad1":87599,
    "squad2":130319,
}
task_to_number_dev = {
    "mnli_m":9816,
    "mnli_mm":9833,
    "qnli":5464,
    "qqp":40431,
    "swag":20007,
    "squad1":10570,
    "squad2":11873,
}

for task_name in args.task.split(':'):
    ic(task_name)

for task_name in args.task.split(':'):
    ic(task_name)

    # get visual image names
    visual_input_path = task_to_path[task_name] + ("_dev" if args.eval else "_train")
    file_list = os.listdir(visual_input_path)

    # get textual prompts
    textual_input_path = registered_path["_".join([args.nlu_dataset, task_name, "dev" if args.eval else "train"])]
    tsv_file = open(textual_input_path)
    dev_list = [[]] + list(csv.reader(tsv_file, delimiter="\t", quoting=csv.QUOTE_NONE))

    if task_name in ["wnli", "rte"]:
        file_list.sort(key = lambda x: (int(x[:x.index('_')]), x[x.index('sentence'):[i for i, n in enumerate(x) if n == '_'][1]]))
        img1_features = []
        img2_features = []
        for idx, file in enumerate(file_list):
            sentence_img = Image.open(os.path.join(visual_input_path, file))
            img_f = clip_model.encode_image(preprocess(sentence_img).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            if (idx % 2) == 0:
                img1_features.append(img_f)
            else:
                img2_features.append(img_f)
        torch.save([torch.tensor(img1_features),torch.tensor(img2_features)], os.path.join(save_root_path, str(task_name) + ("_dev" if args.eval else "_train") + ".pt"))
    elif task_name in ["swag", "sts", "mrpc", "qqp", "qnli", "mnli", "mnli_m", "mnli_mm"]:
        img1_features = []
        img2_features = []
        idx_number = task_to_number_dev[task_name] if args.eval else task_to_number[task_name]
        for idx in range(idx_number):
            file1 = str(idx+1) + "_sentence1.png"
            file2 = str(idx+1) + "_sentence2.png"
            sentence_img1 = Image.open(os.path.join(visual_input_path, file1))
            sentence_img2 = Image.open(os.path.join(visual_input_path, file2))
            img_f1 = clip_model.encode_image(preprocess(sentence_img1).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            img_f2 = clip_model.encode_image(preprocess(sentence_img2).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            img1_features.append(img_f1)
            img2_features.append(img_f2)
        torch.save([torch.tensor(img1_features),torch.tensor(img2_features)], os.path.join(save_root_path, str(task_name) + ("_dev" if args.eval else "_train") + ".pt"))
    elif task_name in ["squad1", "squad2"]:
        img1_features = []
        idx_number = task_to_number_dev[task_name] if args.eval else task_to_number[task_name]
        for idx in range(idx_number):
            file = str(idx+1) + "_question.png"
            sentence_img = Image.open(os.path.join(visual_input_path, file))
            img_f = clip_model.encode_image(preprocess(sentence_img).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            img1_features.append(img_f)
        torch.save([torch.tensor(img1_features)], os.path.join(save_root_path, str(task_name) + ("_dev" if args.eval else "_train") + ".pt"))
    elif task_name in ["sst", "cola"]:
        file_list.sort(key = lambda x: int(x[:x.index('_')]))
        img1_features = []
        txt1_features = []

        for idx, file in enumerate(file_list):
            # visual feature
            sentence_img = Image.open(os.path.join(visual_input_path, file))
            img_pref = clip_model.encode_image(preprocess(sentence_img).unsqueeze(0).to("cuda:0")).cpu()
            img_f = clip_model.encode_image(preprocess(sentence_img).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            img1_features.append(img_f)

            # textual feature
            row_idx = int(file[:file.index('_')])
            row = dev_list[row_idx]
            txt = row[0].replace("\"", "").replace(":", "")
            txt_pref = clip_model.encode_text(clip.tokenize(txt).to("cuda:0")).cpu()
            txt_f = clip_model.encode_text(clip.tokenize(txt).to("cuda:0")).cpu().tolist()[0]
            txt1_features.append(txt_f)

        torch.save([torch.tensor(img1_features)], os.path.join(save_root_path, str(task_name) + ("_dev" if args.eval else "_train") + ".pt")) # [Tensor(# samples, feature dim 255)]
        print("visual feature saved")
        torch.save([torch.tensor(txt1_features)], os.path.join(save_root_path, str(task_name) + ("_textual_dev" if args.eval else "_textual_train") + ".pt")) # [Tensor(# samples, feature dim 255)]
        print("textual feature saved")
    else:
        img1_features = []
        for idx, file in enumerate(file_list):
            sentence_img = Image.open(os.path.join(visual_input_path, file))
            img_f = clip_model.encode_image(preprocess(sentence_img).unsqueeze(0).to("cuda:0")).cpu().tolist()[0]
            img1_features.append(img_f)
