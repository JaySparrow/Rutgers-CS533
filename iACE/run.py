import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from transformers import (
    glue_processors as processors,
    glue_output_modes as output_modes,
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    glue_compute_metrics as compute_metrics,
)

import logging
from tqdm import tqdm
from icecream import ic

from model.cross_modal_encoder import CrossModalEncoder
from data import load_data

## logger
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
## set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# prepare optimizers (with weight decay) and schedulers (with linear warmup)
def prepare_optimizer_and_scheduler(model, total_steps, params: dict):
    no_decay = ["bias", "LayerNorm.weight"]
    model_params = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': params['weight_decay'],
        }, # parameters on which weight decay applies
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        } # parameters on which NO weight decay applies
    ]
    optimizer = AdamW(model_params, lr=params['lr'], eps=params['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(total_steps * params['warmup_ratio']), num_training_steps=total_steps
    )

    return optimizer, scheduler

def train(train_dataset, eval_dataset, tokenizer, model, params: dict, model_langvis=None):
    global logger
    # checkpoint path
    best_ckpt_path = os.path.join(params['output_root'], 'best')

    # data
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=params['batch_size'])

    # total timesteps
    total_steps = len(train_dataloader) * params['num_train_epochs']

    # prepare optimizers (with weight decay) and schedulers (with linear warmup)
    # vlm
    optimizer, scheduler = prepare_optimizer_and_scheduler(model, total_steps, params)
    # cme
    if model_langvis:
        optimizer_langvis, scheduler_langvis = prepare_optimizer_and_scheduler(model_langvis, total_steps, params)
    
    # training
    logger.info("====== Training ======")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", params['num_train_epochs'])
    logger.info("  Total train batch size = %d", params['batch_size'])
    logger.info("  Total optimization steps = %d", total_steps)

    if model_langvis: model_langvis.zero_grad()
    else: model.zero_grad()

    best_eval_acc, best_eval_preds, best_eval_labels = 0., None, None
    set_seed(params['seed'])
    for epoch in tqdm(range(1, params['num_train_epochs']+1), desc="Epoch"):
        epoch_total_loss = 0.

        # train models by turns
        is_langvis_turn = (epoch % 2 == 0) and (epoch <= params['num_train_epochs_langvis'])
        if model_langvis and is_langvis_turn: 
            model_langvis.train()
            model.eval()
        else: 
            model.train()
            if model_langvis: model_langvis.eval()
        # model.train()
        # if model_langvis: model_langvis.train()

        for step, batch in tqdm(enumerate(train_dataloader), desc="Train Batch"):

            # prepare batch dict
            batch = tuple(b.to(device) for b in batch)
            batch_dict = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels': batch[3]
            }

            # vlm forward pass
            outputs = model(**batch_dict)
            loss, logits = outputs[:2]

            # cme forward pass (bidirectional)
            if model_langvis:
                cme_logits1, cme_loss1 = model_langvis(torch.tensor(batch[4]).cuda(device), torch.tensor(batch[5]).cuda(device), labels=batch[3])
                cme_logits2, cme_loss2 = model_langvis(torch.tensor(batch[5]).cuda(device), torch.tensor(batch[4]).cuda(device), labels=batch[3])
                loss += cme_loss1 + cme_loss2

            # backward pass
            loss.backward()

            # record total training loss
            epoch_total_loss += loss.item()

            if model_langvis and is_langvis_turn: 
                # clip gradients
                torch.nn.utils.clip_grad_norm_(model_langvis.parameters(), params['max_grad_norm'])
                # step optimizer and scheduler
                optimizer_langvis.step()
                scheduler_langvis.step()
                # zero gradients
                model_langvis.zero_grad()
            else:
                # clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
                # step optimizer and scheduler
                optimizer.step()
                scheduler.step()
                # zero gradients
                model.zero_grad()

            # ## cme
            # if model_langvis:
            #     # clip gradients
            #     torch.nn.utils.clip_grad_norm_(model_langvis.parameters(), params['max_grad_norm'])
            #     # step optimizer and scheduler
            #     optimizer_langvis.step()
            #     scheduler_langvis.step()
            #     # zero gradients
            #     model_langvis.zero_grad()
            # ## vlm
            # # clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
            # # step optimizer and scheduler
            # optimizer.step()
            # scheduler.step()
            # # zero gradients
            # model.zero_grad()

        epoch_avg_loss = epoch_total_loss / len(train_dataloader)

        ## Evaluation
        if params['do_eval']:
            eval_avg_loss, eval_acc, eval_preds, eval_labels = evaluate(eval_dataset, tokenizer, model, params, model_langvis, prefix="Epoch " + str(epoch))
            log_info = "  Epoch %d | train loss = %.3f | eval loss = %.3f | eval acc = %.3f"%(epoch, epoch_avg_loss, eval_avg_loss, eval_acc)

            ## Save best
            if eval_acc > best_eval_acc:
                best_eval_acc, best_eval_preds, best_eval_labels = eval_acc, eval_preds, eval_labels
                if not os.path.exists(best_ckpt_path): os.makedirs(best_ckpt_path)
                # save models
                model.save_pretrained(best_ckpt_path)
                tokenizer.save_pretrained(best_ckpt_path)
                if model_langvis: torch.save(model_langvis.state_dict(), os.path.join(best_ckpt_path, 'model_langvis.pth'))
                # save parameters
                torch.save(params, os.path.join(best_ckpt_path, 'training_params.bin'))
                # save evaluation results
                output_eval_file = os.path.join(best_ckpt_path, 'best_eval_results.txt')
                with open(output_eval_file, 'w') as writer:
                    writer.write("acc = %s\n" % (str(float(best_eval_acc))))
                    writer.write("\n".join(map(str, (best_eval_preds==best_eval_labels).flatten())))

                log_info += "<---------- saved"
            
            # log
            logger.info(log_info)

def evaluate(dataset, tokenizer, model, params: dict, model_langvis=None, prefix: str=""):
    global logger

    # data
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=params['batch_size'])

    # evaluating
    logger.info("=== Evaluating %s===", prefix+" ")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Total eval batch size = %d", params['batch_size'])

    eval_total_loss = 0.
    all_logits = None
    all_labels = None
    model.eval()
    if model_langvis: model_langvis.eval()
    for batch in tqdm(eval_dataloader, desc="Eval Batch"):

        # prepare batch dict
        batch = tuple(b.to(device) for b in batch)
        batch_dict = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'labels': batch[3]
        }

        with torch.no_grad():
            # vlm forward pass
            outputs = model(**batch_dict)
            loss, logits = outputs[:2]

            # cme forward pass (bidirectional)
            if model_langvis:
                cme_logits1, cme_loss1 = model_langvis(torch.tensor(batch[4]).cuda(device), torch.tensor(batch[5]).cuda(device), labels=batch[3])
                cme_logits2, cme_loss2 = model_langvis(torch.tensor(batch[5]).cuda(device), torch.tensor(batch[4]).cuda(device), labels=batch[3])
                loss += cme_loss1 + cme_loss2
                # average logits
                logits = (logits + (cme_logits1 + cme_logits2) / 2) / 2 
            
        eval_total_loss += loss.item()

        # record logits and labels
        if all_logits is None:
            all_logits = logits.detach().cpu().numpy()
            all_labels = batch_dict['labels'].detach().cpu().numpy()
        else:
            all_logits = np.append(all_logits, logits.detach().cpu().numpy(), axis=0) # (# samples, # classes)
            all_labels = np.append(all_labels, batch_dict['labels'].detach().cpu().numpy(), axis=0) # (# samples)

    eval_avg_loss = eval_total_loss / len(eval_dataloader)
    # classification
    all_preds = np.argmax(all_logits, axis=1) # (# samples)
    all_acc = compute_metrics(params['task_name'], all_preds, all_labels)['acc']

    return eval_avg_loss, all_acc, all_preds, all_labels
    
def finetuning(use_langvis: bool=True):
    seed = 9595
    task_name = 'sst-2'
    tokenizer_name = 'bert-base-uncased'
    pretrained_vlm_path = 'snap/vlm/vlm_12L_768H_wiki'
    cache_dir = ""
    max_seq_len=126
    train_percentage=0.1 # 100
    eval_percentage=100

    params = {
        'seed': seed,
        'task_name': task_name,
        'lr': 1e-4,
        'weight_decay': 0.0,
        'adam_epsilon': 1e-6,
        'warmup_ratio': 0.1,
        'batch_size': 32,
        'num_train_epochs': 30, # 30,
        'num_train_epochs_langvis': 5, # 5,
        'max_grad_norm': 1.0,
        'output_root': 'checkpoints/proposed-0.1',
        'do_eval': False,
    }

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set seed
    set_seed(seed)

    # task settings
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # load the config for the pretrained vlm
    config = AutoConfig.from_pretrained(
        pretrained_vlm_path,
        num_labels=num_labels,
        finetuning_task=task_name,
        cache_dir=cache_dir if cache_dir else None,
    )

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=False,
        do_lower_case=True,
        cache_dir=cache_dir if cache_dir else None,
    )

    # load pretrained vlm
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_vlm_path,
        from_tf=False,
        config=config,
        cache_dir=cache_dir if cache_dir else None,
    )
    model.to(device)

    # create cross-modal encoder 
    if task_name in ['sst-2']:
        task_num_classes = 2
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    if use_langvis:
        model_langvis = CrossModalEncoder(
                            num_classes=task_num_classes,
                            loss_fn=loss_func,
                            in_lan_feature_dim=512,
                            in_vis_feature_dim=512,
                            out_fusion_dim=512,
                            dropout_prob=0.1,
                        ).cuda(device)
        model_langvis.to(device)
    else:
        model_langvis = None

    # load dataset
    train_dataset = load_data(tokenizer, task=task_name, max_seq_len=max_seq_len, is_eval=False, percentage=train_percentage)
    eval_dataset = load_data(tokenizer, task=task_name, max_seq_len=max_seq_len, is_eval=True, percentage=eval_percentage)

    # training
    train(train_dataset, eval_dataset, tokenizer, model, params, model_langvis=model_langvis)

    # evaluation
    eval_avg_loss, eval_acc, eval_preds, eval_labels = evaluate(eval_dataset, tokenizer, model, params, model_langvis, prefix="")
    logger.info("final eval_acc = %.3f", eval_acc)

    # save final checkpoint
    final_ckpt_path = os.path.join(params['output_root'], 'final')
    if not os.path.exists(final_ckpt_path): os.makedirs(final_ckpt_path)
    # save models
    model.save_pretrained(final_ckpt_path)
    tokenizer.save_pretrained(final_ckpt_path)
    if model_langvis: torch.save(model_langvis.state_dict(), os.path.join(final_ckpt_path, 'model_langvis.pth'))
    # save parameters
    torch.save(params, os.path.join(final_ckpt_path, 'training_params.bin'))
    # save evaluation results
    output_eval_file = os.path.join(final_ckpt_path, 'eval_results.txt')
    with open(output_eval_file, 'w') as writer:
        writer.write("acc = %s\n" % (str(float(eval_acc))))
        writer.write("\n".join(map(str, (eval_preds==eval_labels).flatten())))

def inference():
    seed = 9595
    task_name = 'sst-2'
    tokenizer_name = 'bert-base-uncased'
    pretrained_path = 'checkpoints/proposed-0.1/final'
    cache_dir = ""
    max_seq_len=126
    eval_percentage=100

    params = {
        'seed': seed,
        'task_name': task_name,
        'lr': 1e-4,
        'weight_decay': 0.0,
        'adam_epsilon': 1e-6,
        'warmup_ratio': 0.1,
        'batch_size': 32,
        'num_train_epochs': 30,
        'num_train_epochs_langvis': 5,
        'max_grad_norm': 1.0,
    }

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set seed
    set_seed(seed)

    # task settings
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_path,
        use_fast=False,
        # do_lower_case=True,
        # cache_dir=cache_dir if cache_dir else None,
    )

    # load pretrained vlm
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_path,
        # from_tf=False,
        # config=config,
        # cache_dir=cache_dir if cache_dir else None,
    )
    model.to(device)

    # create cross-modal encoder 
    if task_name in ['sst-2']:
        task_num_classes = 2
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError
    if os.path.exists(os.path.join(pretrained_path, 'model_langvis.pth')):
        model_langvis = CrossModalEncoder(
                            num_classes=task_num_classes,
                            loss_fn=loss_func,
                            in_lan_feature_dim=512,
                            in_vis_feature_dim=512,
                            out_fusion_dim=512,
                            dropout_prob=0.1,
                        ).cuda(device)
        # model_langvis.to(device)
        model_langvis.load_state_dict(torch.load(os.path.join(pretrained_path, 'model_langvis.pth')))
    else:
        model_langvis = None

    # load dataset
    eval_dataset = load_data(tokenizer, task=task_name, max_seq_len=max_seq_len, is_eval=True, percentage=eval_percentage, tokenizer_name=tokenizer_name)

    # evaluating
    eval_avg_loss, eval_acc, eval_preds, eval_labels = evaluate(eval_dataset, tokenizer, model, params, model_langvis, prefix="")
    logger.info("  eval loss = %.5f | eval acc = %.5f"%(eval_avg_loss, eval_acc))

if __name__ == '__main__':
    finetuning()
    # inference()