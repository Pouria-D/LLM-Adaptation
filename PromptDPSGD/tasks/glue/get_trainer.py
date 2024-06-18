import logging

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPT2Tokenizer,
)

from model.utils import get_model, TaskType
from private_transformers.examples.classification.src.trainer import Trainer as TrainerPrivateTransformers
from tasks.glue.dataset import GlueDataset
from training.trainer_base import BaseTrainer
from training.trainer_base import BaseTrainerOpacus

logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args, _, privacy_args, auxiliary_args = args

    if model_args.model_name_or_path == 'gpt2':
        # Get model's tokenizer.
        print('Loading tokenizer for gpt2...', flush=True)
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_args.model_name_or_path)
        # default to left padding
        tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
        )

    dataset = GlueDataset(tokenizer, data_args, training_args)

    if not dataset.is_regression:
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

    ### fix_bert=True for LastLayer Tuning
    model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config)

    if model_args.model_name_or_path == 'gpt2':
        # resize model embedding to match new tokenizer
        model.resize_token_embeddings(len(tokenizer))

        # fix model padding token id
        model.config.pad_token_id = model.config.eos_token_id

    model = model.to(training_args.device)
    print('model: ', model)

    if training_args.training_type == 'private':
        print('Use opacus for private training.')
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

        # data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

        # trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        #     args=training_args,
        #     model=model,
        #     train_dataset=dataset.train_dataset,
        #     eval_dataset=dataset.eval_dataset,
        #     privacy_args=privacy_args,
        #     data_collator=dataset.data_collator,
        #     compute_metrics=dataset.compute_metrics,
        #     tokenizer=tokenizer,
        # )
        print('privacy_engine: ', training_args.privacy_engine)
        if training_args.privacy_engine == "dp_transformers":
            trainer = BaseTrainerOpacus(
                args=training_args,
                model=model,
                train_dataset=dataset.train_dataset,
                eval_dataset=dataset.eval_dataset,
                privacy_args=privacy_args,
                data_collator=dataset.data_collator,
                compute_metrics=dataset.compute_metrics,
                tokenizer=tokenizer,
            )
        elif training_args.privacy_engine == "private_transformers":
            # trainer = BaseTrainer(
            #     model=model,
            #     args=training_args,
            #     train_dataset=dataset.train_dataset,
            #     eval_dataset=dataset.eval_dataset,
            #     compute_metrics=dataset.compute_metrics,
            #     tokenizer=tokenizer,
            #     data_collator=dataset.data_collator,
            # )

            trainer = TrainerPrivateTransformers(
                model=model,
                args=training_args,
                model_args=model_args,
                privacy_args=privacy_args,
                auxiliary_args=auxiliary_args,
                train_dataset=dataset.train_dataset,
                eval_dataset=dataset.eval_dataset,
                tokenizer=tokenizer,
                compute_metrics=dataset.compute_metrics,
            )
        else:
            raise Exception(f"Unsupported privacy engine: {training_args.privacy_engine}.")

    elif training_args.training_type == 'public':
        # Initialize our Trainer
        trainer = BaseTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset,
            eval_dataset=dataset.eval_dataset,
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
        )
    else:
        raise Exception(f"Unsupported training_type: {training_args.training_type}.")

    return trainer, None
