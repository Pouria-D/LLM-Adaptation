import logging
import os
import random
import sys
import torch
import dp_transformers

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

from model.utils import get_model, TaskType
from tasks.superglue.dataset import SuperGlueDataset
from training.trainer_base import BaseTrainer
from training.trainer_exp import ExponentialTrainer

logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args, _, privacy_args = args

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset = SuperGlueDataset(tokenizer, data_args, training_args)

    if training_args.do_train:
        for index in random.sample(range(len(dataset.train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {dataset.train_dataset[index]}.")

    if not dataset.multiple_choice:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )

    if not dataset.multiple_choice:
        model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)
    else:
        model = get_model(model_args, TaskType.MULTIPLE_CHOICE, config, fix_bert=True)

    print('model: ', model)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, eps=1e-8)

    if training_args.training_type == 'private':
        # print('Use opacus')
        # from opacus import PrivacyEngine
        # privacy_engine = PrivacyEngine()
        #
        # model.train()
        #
        # model, optimizer, train_dataloader = privacy_engine.make_private_with_epsilon(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=dataset.train_dataset,
        #     target_delta=model_args.dp_delta,
        #     target_epsilon=model_args.dp_epsilon,
        #     epochs=training_args.num_train_epochs,
        #     max_grad_norm=model_args.dp_max_grad_norm,
        # )

        tokenizer.pad_token = -100  # Set a dummy pad token we don't use it anyway

        # Tokenize data
        with training_args.main_process_first(desc="tokenizing dataset"):
            dataset = dataset.map(
                lambda batch: tokenizer(batch['content'], padding="max_length", truncation=True,
                                        max_length=args.model.sequence_len),
                batched=True, num_proc=8, desc="tokenizing dataset", remove_columns=dataset.column_names['train']
            )

        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset.train_dataset,
            eval_dataset=dataset.eval_dataset,
            data_collator=data_collator,
            privacy_args=privacy_args,
        )

    elif training_args.training_type == 'public':
        # Initialize the standard Trainer without opacus
        trainer = BaseTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            test_key=dataset.test_key,
            optimizers=(optimizer, None),
        )
    else:
        raise Exception(f"Unsupported training_type: {training_args.training_type}.")

    return trainer, None
