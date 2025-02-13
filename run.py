import re
import torch

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer


print("CUDA GPU backend", torch.cuda.is_available())

ds = load_dataset("qiaojin/PubMedQA", "pqa_unlabeled")

print(ds)


SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and given a context, the Assistant solves it. The assistant "
    "first thinks about the context, then runs a reasoning process in the mind and finally provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

def make_conversation(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "".join(example["context"]['contexts'])},
            {"role": "user", "content": example["question"]},
        ],
    }

ds = ds.map(make_conversation)
train_dataset = ds.remove_columns(['pubid', 'question', 'context'])


model_id = "Qwen/Qwen2-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()


def format_reward(completions, **kwargs):
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    rewards_list = [1.0 if match else 0.0 for match in matches]
    return [1.0 if match else 0.0 for match in matches]



# Load a pretrained SBERT model
encoding_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_similarity(paragraph1, paragraph2):
    embedding1 = encoding_model.encode(paragraph1, convert_to_tensor=True)
    embedding2 = encoding_model.encode(paragraph2, convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
    return similarity

def similary_reward(completions, **kwargs):
    long_answers = kwargs["long_answer"]
    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, long_answer in zip(completion_contents, long_answers):
        similarity = get_similarity(content, long_answer)
        if similarity > 0.9:
            rewards.append(1.)
        elif similarity > 0.7:
            rewards.append(0.5)
        elif similarity > 0.5:
            rewards.append(0.0)
        else:
            rewards.append(-1.0)
    return rewards


# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    output_dir="Qwen2-0.5B-GRPO-test",
    learning_rate=1e-5,
    remove_unused_columns=False, # to access the solution column in accuracy_reward
    gradient_accumulation_steps=16,
    num_train_epochs=1,
    bf16=True,

    # Parameters that control de data preprocessing
    max_completion_length=64, # default: 256
    num_generations=4, # default: 8
    max_prompt_length=128, # default: 512

    # Parameters related to reporting and saving
    report_to=["tensorboard"],
    logging_steps=10,
    push_to_hub=False,
    save_strategy="steps",
    save_steps=10,
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[format_reward, similary_reward],
    args=training_args,
    train_dataset=train_dataset['train']
)

trainer.train()
