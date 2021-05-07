### The baseline models and baseline+CE(context extension framework) models

import argparse
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMultipleChoice,
    BertTokenizer,
    RobertaConfig,
    RobertaForMultipleChoice,
    RobertaTokenizer,
    XLNetConfig,
    XLNetForMultipleChoice,
    XLNetTokenizer,
    AlbertConfig,
    AlbertForMultipleChoice,
    AlbertTokenizer,
    get_linear_schedule_with_warmup,
)
from utils_multiple_choice import convert_examples_to_features, processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForMultipleChoice, BertTokenizer),
    "xlnet": (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
    "roberta": (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer),
    "albert": (AlbertConfig, AlbertForMultipleChoice, AlbertTokenizer),
}


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ", # + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run test on the test set")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument('--adam_betas', default='(0.9, 0.999)', type=str, help='betas for Adam optimizer')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_clip_grad_norm", action="store_true", help="whether not to clip grad norm")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup over warmup ratios.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--ques_type_before",
        type=int,
        default=1,
        help="Whether to place question type before question",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")

    parser.add_argument(
        "--whether_extend_context",
        action="store_true",
        help="Whether to utilize the logic-driven context extension framework",
    )
    parser.add_argument(
        "--extended_context_version",
        type=int,
        default=1,
        help="The version of extended context",
    )
    parser.add_argument(
        "--negative_context_version",
        type=int,
        default=1,
        help="The version of negative context",
    )
    parser.add_argument(
        "--negative_entend_context_version",
        type=int,
        default=1,
        help="The version of negative entend context",
    )

    return parser.parse_args()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_questype_extn".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir, args.extended_context_version)
        elif test:
            examples = processor.get_test_examples(args.data_dir, args.extended_context_version)
        else:
            examples = processor.get_train_examples(args.data_dir, args.extended_context_version)
        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            ques_type_before=args.ques_type_before,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            whether_extend_context=args.whether_extend_context,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def train(args, train_dataset, model, tokenizer, test_dataset=None):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        str_list = str(args.output_dir).split('/')
        tb_log_dir = os.path.join('summaries', str_list[-1])
        tb_writer = SummaryWriter(tb_log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    exec('args.adam_betas = ' + args.adam_betas)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, betas=args.adam_betas, eps=args.adam_epsilon)
    assert not ((args.warmup_steps > 0) and (args.warmup_proportion > 0)), "--only can set one of --warmup_steps and --warm_ratio "
    if args.warmup_proportion > 0:
        args.warmup_steps = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("************************* Running training *************************")
    logger.info("Num examples = %d", len(train_dataset))
    logger.info("Num Epochs = %d", args.num_train_epochs)
    logger.info("Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
        )
    # logger.info("Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("Total optimization steps = %d", t_total)

    val_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True, test=False)

    def evaluate_model(train_preds, train_label_ids, tb_writer, args, model, tokenizer, best_steps, best_dev_acc, val_dataset):
        train_preds = np.argmax(train_preds, axis=1)
        train_acc = simple_accuracy(train_preds, train_label_ids)
        train_preds = None
        train_label_ids = None
        results = evaluate(args, model, tokenizer, val_dataset)
        logger.info(
            "dev acc: %s, loss: %s, global steps: %s",
            str(results["eval_acc"]),
            str(results["eval_loss"]),
            str(global_step),
        )
        tb_writer.add_scalar("training/acc", train_acc, global_step)
        for key, value in results.items():
            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
        if results["eval_acc"] > best_dev_acc:
            best_dev_acc = results["eval_acc"]
            best_steps = global_step
            logger.info("!!!!!!!!!!!!!!!!!!!! achieve BEST dev acc: %s at global step: %s",
                        str(best_dev_acc),
                        str(best_steps)
                        )

            # save best dev acc model
            output_dir = args.output_dir
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            logger.info("Current local rank %s", args.local_rank)
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_vocabulary(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)
            txt_dir = os.path.join(output_dir, 'best_dev_results.txt')
            with open(txt_dir, 'w') as f:
                rs = 'global_steps: {}; dev_acc: {}'.format(global_step, best_dev_acc)
                f.write(rs)
                tb_writer.add_text('best_results', rs, global_step)

        logger.info("current BEST dev acc: %s at global step: %s",
                    str(best_dev_acc),
                    str(best_steps)
                    )

        return train_preds, train_label_ids, train_acc, best_steps, best_dev_acc

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc = 0.0
    best_steps = 0
    train_preds = None
    train_label_ids = None
    model.zero_grad()
    set_seed(args)  # Added here for reproductibility
    for epoch_index in range(int(args.num_train_epochs)):
        logger.info('')
        logger.info('%s Epoch: %d %s', '*'*50, epoch_index, '*'*50)
        for step, batch in enumerate(train_dataloader):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2]
                if args.model_type in ["bert", "xlnet", "albert"]
                else None,  # XLM, Roberta don't use segment_ids
                "labels": batch[3],
            }
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits = outputs[1]

            ################# work only gpu = 1 ######################
            if train_preds is None:
                train_preds = logits.detach().cpu().numpy()
                train_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                train_preds = np.append(train_preds, logits.detach().cpu().numpy(), axis=0)
                train_label_ids = np.append(train_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            ###########################################################

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if not args.no_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if not args.no_clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()

            if step % 20 == 0:
                logger.info("********** Iteration %d: current loss: %s", step, str(round(loss.item(), 4)),)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                # optimizer.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # if (args.local_rank == -1 and args.evaluate_during_training):  # Only evaluate when single GPU otherwise metrics may not average well
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        train_preds, train_label_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds, train_label_ids, tb_writer, args, model, tokenizer, best_steps, best_dev_acc, val_dataset)
                        tb_writer.add_scalar("training/lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("training/loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logger.info(
                            "Average loss: %s, average acc: %s at global step: %s",
                            str((tr_loss - logging_loss) / args.logging_steps),
                            str(train_acc),
                            str(global_step),
                        )
                        logging_loss = tr_loss

                # if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                #     save_model(args, model, tokenizer)
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break

    if args.local_rank in [-1, 0] and train_preds is not None:
        train_preds, train_label_ids, train_acc, best_steps, best_dev_acc = evaluate_model(train_preds, train_label_ids, tb_writer, args, model, tokenizer, best_steps, best_dev_acc, val_dataset)
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps


def evaluate(args, model, tokenizer, val_dataset=None, prefix="", test=False):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = val_dataset

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("************************* Running evaluation {} *************************".format(prefix))
        logger.info("Num examples = %d", len(eval_dataset))
        logger.info("Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        model.eval()
        for batch in eval_dataloader:
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2]
                    if args.model_type in ["bert", "xlnet", "albert"]
                    else None,  # XLM don't use segment_ids
                    "labels": batch[3],
                }
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)

        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)

        logger.info("***** Eval results {} *****".format(str(prefix) + "----is test:" + str(test)))
        if test:
            for key in sorted(result.keys()):
                logger.info("%s = %s", key, str(result[key]))

    if test:
        return results, preds
    else:
        return results


def main():
    args = init_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
        logger.warning('local_rank: %s, gpu_num: %s', torch.distributed.get_rank(), torch.cuda.device_count(),)
        args.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        args.n_gpu = 1
    args.device = device

    # set random seed
    set_seed(args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # logger.info('n_gpu: %s, world_size: %s', args.n_gpu, torch.distributed.get_world_size())

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.whether_extend_context:
        # add special token [EXT]
        special_tokens_dict = {'additional_special_tokens': ['<ext>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    test_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False, test=True)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)

        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer, test_dataset)
        logger.info("global_step = %s, average loss = %s", global_step, tr_loss)

    # Test

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            checkpoint_dir = args.model_name_or_path
        if args.evaluate_during_training:
            checkpoint_dir = os.path.join(args.output_dir)
        logger.info('load checkpoint_dir: %s', checkpoint_dir)
        logger.info('current local rank: %s', args.local_rank)
        if best_steps:
            logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)

        model = model_class.from_pretrained(checkpoint_dir)
        model.to(args.device)
        result, preds = evaluate(args, model, tokenizer, test_dataset, test=True)
        np.save(os.path.join(args.output_dir, "test_preds.npy" if args.output_dir is not None else "test_preds.npy"), preds)


if __name__ == "__main__":
    main()
