import json
import logging
import os
import sys
import time
from datetime import datetime

import datasets
import numpy as np
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from arguments import get_args
from private_transformers.privacy_engine import PrivacyEngine
from tasks.utils import *

os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)


def get_checkpoint(resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None
    print("resuem for checkpoint:", resume_from_checkpoint)
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    return checkpoint


def show_train_result(train_result):
    key = "metrics"
    if hasattr(trainer, key):
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

    key = 'save_state'
    if hasattr(trainer, key) and callable(getattr(trainer, key)):
        trainer.save_state()

    key = 'log_best_metrics'
    if hasattr(trainer, key) and callable(getattr(trainer, key)):
        trainer.log_best_metrics()

    key = 'get_prv_epsilon'
    if hasattr(trainer, key) and callable(getattr(trainer, key)):
        eps_prv = trainer.get_prv_epsilon()
    else:
        eps_prv = 0

    key = 'get_rdp_epsilon'
    if hasattr(trainer, key) and callable(getattr(trainer, key)):
        eps_rdp = trainer.get_rdp_epsilon()
    else:
        eps_rdp = 0

    key = 'log'
    if hasattr(trainer, key) and callable(getattr(trainer, key)):
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })


def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = get_checkpoint(resume_from_checkpoint=resume_from_checkpoint, last_checkpoint=last_checkpoint)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    show_train_result(train_result=train_result)


def train_private_transformer(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    # checkpoint = get_checkpoint(resume_from_checkpoint=resume_from_checkpoint, last_checkpoint=last_checkpoint)

    # params that require the grad
    named_params = [(name, param) for name, param in trainer.model.named_parameters() if param.requires_grad]
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
         'weight_decay': training_args.weight_decay},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = trainer.optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    if training_args.lr_decay == 'yes':
        print('Apply default linear decay.')
        training_setup = trainer.get_training_setup()
        t_total = training_setup["t_total"]
        # `trainer.optimizer` is not None here, so no optimizer is created.
        trainer.create_optimizer_and_scheduler(num_training_steps=t_total)
    else:
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(trainer.optimizer, lambda _: 1.)

    total_train_batch_size = training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size
    privacy_engine = PrivacyEngine(
        module=trainer.model,
        batch_size=total_train_batch_size,
        sample_size=len(trainer.train_dataset),
        epochs=training_args.num_train_epochs,
        max_grad_norm=privacy_args.per_sample_max_grad_norm,
        noise_multiplier=privacy_args.noise_multiplier,
        target_epsilon=privacy_args.target_epsilon,
        target_delta=privacy_args.target_delta,
        accounting_mode=privacy_args.accounting_mode,
        clipping_mode=privacy_args.clipping_mode,
        skip_checks=True,
    )
    # Originally, it could have been null.
    privacy_args.noise_multiplier = privacy_engine.noise_multiplier
    privacy_args.target_delta = privacy_engine.target_delta
    print('privacy_engine.noise_multiplier: ', privacy_engine.noise_multiplier)
    print('privacy_engine.target_delta: ', privacy_engine.target_delta)

    print('privacy_args: ')
    print(json.dumps(privacy_args.__dict__, indent=4))
    privacy_engine.attach(optimizer)

    # Training
    train_result = trainer.train()
    trainer.save_model()

    show_train_result(train_result=train_result)


def evaluate(trainer):
    logger.info("*** Evaluate ***")
    # Check trainer prediction step - no labels.
    metrics = trainer.evaluate()

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    return metrics


def predict(trainer, predict_dataset=None):
    if predict_dataset is None:
        logger.info("No dataset is available for testing")

    elif isinstance(predict_dataset, dict):

        for dataset_name, d in predict_dataset.items():
            logger.info("*** Predict: %s ***" % dataset_name)
            predictions, labels, metrics = trainer.predict(d, metric_key_prefix="predict")
            predictions = np.argmax(predictions, axis=2)

            trainer.log_metrics("predict", metrics)
            trainer.save_metrics("predict", metrics)

    else:
        logger.info("*** Predict ***")
        predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")
        predictions = np.argmax(predictions, axis=2)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


def write_metrics(metrics, elapsed_time=0):
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    print(data_args.dataset_name)
    print('metrics file: ', f"tune_params_lr_grad_epochs_{data_args.dataset_name}.csv")
    with open(f"tune_params_lr_grad_epochs_{data_args.dataset_name}.csv", "a") as writer:
        writer.write(
            f"{training_args.training_type},{training_args.privacy_engine},{model_args.method_type},{data_args.dataset_name},{model_args.model_name_or_path},{training_args.per_device_train_batch_size},{training_args.learning_rate},{privacy_args.target_epsilon},{privacy_args.per_sample_max_grad_norm},{training_args.num_train_epochs},{model_args.pre_seq_len},{metrics['eval_loss']},{metrics['eval_accuracy']},{elapsed_time},{elapsed_time / 3600},{current_time},{training_args.gradient_accumulation_steps},{privacy_args.target_delta},{training_args.training_type}\n")


def set_mapping_for_infilling(data_args):
    # from: https://arxiv.org/abs/2110.05679
    # TODO: Hacky mapping creation. Refactor this in the future.
    #  Currently gets replace if mapping_id and mapping_path is set.
    if data_args.dataset_name == "sst2":
        data_args.mapping = "{'0':'terrible','1':'great'}"
    elif data_args.dataset_name == "mnli":
        data_args.mapping = "{'contradiction': 'no', 'entailment': 'yes', 'neutral': 'maybe'}"
    elif data_args.dataset_name == "qnli":
        data_args.mapping = "{'not_entailment': 'no', 'entailment': 'yes'}"
    elif data_args.dataset_name == "qqp":
        data_args.mapping = "{'1': 'yes', '0': 'no'}"  # 1 -- equivalent, 0 -- not equivalent.
    else:
        raise ValueError(f"Unknown task: {data_args.task_name}")


if __name__ == '__main__':
    args = get_args()
    print('args: ', args)
    model_args, data_args, training_args, _, privacy_args, _ = args

    # set_mapping_for_infilling(data_args=data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if data_args.task_name.lower() == "glue":
        assert data_args.dataset_name.lower() in GLUE_DATASETS
        from tasks.glue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "superglue":
        assert data_args.dataset_name.lower() in SUPERGLUE_DATASETS
        from tasks.superglue.get_trainer import get_trainer

    elif data_args.task_name.lower() == "ner":
        assert data_args.dataset_name.lower() in NER_DATASETS
        from tasks.ner.get_trainer import get_trainer

    elif data_args.task_name.lower() == "srl":
        assert data_args.dataset_name.lower() in SRL_DATASETS
        from tasks.srl.get_trainer import get_trainer

    elif data_args.task_name.lower() == "qa":
        assert data_args.dataset_name.lower() in QA_DATASETS
        from tasks.qa.get_trainer import get_trainer
    else:
        raise NotImplementedError(
            'Task {} is not implemented. Please choose a task from: {}'.format(data_args.task_name, ", ".join(TASKS)))

    set_seed(training_args.seed)

    trainer, predict_dataset = get_trainer(args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # if training_args.do_eval:
    #     metrics = evaluate(trainer)
    #     # write_metrics(metrics=metrics)

    elapsed_time = 0
    if training_args.do_train:
        start = time.time()
        if training_args.training_type == 'private':
            if training_args.privacy_engine == 'private_transformers':
                train_private_transformer(trainer=trainer, resume_from_checkpoint=training_args.resume_from_checkpoint,
                                          last_checkpoint=last_checkpoint)
            elif training_args.privacy_engine == 'dp_transformers':
                train(trainer, training_args.resume_from_checkpoint, last_checkpoint)
            else:
                raise Exception(f"Unsupported privacy engine: {training_args.privacy_engine}.")
        elif training_args.training_type == 'public':
            train(trainer, training_args.resume_from_checkpoint, last_checkpoint)
        else:
            raise Exception(f"Unsupported training_type: {training_args.training_type}.")
        stop = time.time()
        elapsed_time = stop - start

    if training_args.do_eval:
        metrics = evaluate(trainer)
        write_metrics(metrics=metrics, elapsed_time=elapsed_time)

    if training_args.do_predict:
        predict(trainer, predict_dataset)
