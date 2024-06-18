# Tuning Methods Comparision

In this project we have compare different method of fineTuning transformers on specific tasks.

This repository is a customized clone of https://github.com/cleverhans-lab/PrivatePrompts whihc has been customized to complete our comparision.

**Methods:**
* Soft Prompt 
* Prefix
* Lora
* Full FineTune
* Last Layer 
* Lora + Prompt
* Lora + Prefix

**Tasks:**
* Glue: sst2, qnli, qqp, mnli

**Model:**
* prajjwal1/bert-tiny


The models used are in the `model.sequence_classification` module and the specific classes used in our experiments in the paper are:

- BertForSequenceClassification

# How To Run

To install the required packages navigate to `install.txt`.

An example of how to run the code is in `run.sh` file.

and there have been additional packages comapre to main implementation which have been installed in main Notebook.

**SoftPrompt, Prefix, Finetune:**

To run these methods, just pass the method name through `method_type` argument.

**Last Layer:**

This method is a simple finetune in a way that all transformer( In this case bert) papermeters have been freezed and we have just last layer (classification layer) trainable.

So for this tuning you should pass `fix_bert=True` to get_model function when calling model.

~ tasks/glue/get_trainer.py :
```python
   model = get_model(model_args, TaskType.SEQUENCE_CLASSIFICATION, config, fix_bert=True)
```

**LoRA (and its combiniations):**

To implement Low Rank Adaption method, we have implemented a LinearWithLora class that can be applied on every desired base model by calling apply_lora function.

so to fineTune each method with LoRA-enable mood, uncomment `apply_lora` on the model:

~ model/sequence_classifciation.py:
```python
#LoRA:
Class BertForSequenceClassification

#LoRA + Prefix:
Class BertPrefixForSequenceClassification

#LoRA + Prompt:
Class BertPromptForSequenceClassification

###### Apply Lora
self.bert = apply_lora(self.bert)
#####
```
# Privacy-Enabled Tuning

To achive Differential Privacy Tuning to avoid release information about private downstream dataset, we want to use DPSGD algorithm to clipp base gradients and add appropriate noise (in gaussian distribution) to the gradient

to this aim, there have been developed some libraries such as OPACUS and intermediate libraries to work with that based on target model which is transformer here. like `dp-transformer` which is develped by microsoft.

We have used `private-transformers` library to enable dp training with `clipping norm=1.0` and `target-epsilon=8` 

Point: As we wanted to develop the main architechture of implementation of Paper, there was some conflicts and missunderstaindings between transforemers libraries and modification which has been made by microsoft on dp-transformers to `Trainer` class and we couldn't find approprate version of libraries to satisfy the required `kwargs`.
The main privacy engine in this work is from:  `private_transformers.privacy_engine.PrivacyEngine`.

# Acknowledgement
The code in private_transformers directory comes from: https://github.com/lxuechen/private-transformers and is in this submission only to enable the run of our code.

This code in dp_transformers comes from https://github.com/microsoft/dp-transformers and is in this submission only to enable the run of our code.




