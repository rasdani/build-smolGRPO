import os

from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from reward import compute_score

os.environ["WANDB_PROJECT"] = "smolGRPO"

def reward_func(completions, ground_truth, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        content = completion[0]["content"]
        reward = compute_score(content, gt)
        rewards.append(reward["score"])
    return rewards

def calc_gradient_accumulation_steps(num_gpus, per_device_train_batch_size, target_batch_size=1024):
    return int(target_batch_size / (per_device_train_batch_size * num_gpus))

model_name = "Qwen/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset("hkust-nlp/SimpleRL-Zoo-Data", 
                      data_files="simplelr_qwen_level1to4/train.parquet",
                      split="train")
train_dataset = dataset.rename_column(original_column_name="gt_answer", new_column_name="ground_truth")

num_gpus = 2
# per_device_train_batch_size = 4
per_device_train_batch_size = 8
run_name = "simple_rl_zoo_" + model_name.split("/")[-1].lower()


training_args = GRPOConfig(
    output_dir=f"outputs/{run_name}",
    run_name=run_name,
    learning_rate=5e-7,
    learning_rate_scheduler="constant",
    num_train_epochs=1,
    bf16=True,
    num_iterations=1,
    beta=0.0001,
    epsilon=0.2,
    max_prompt_length=1024,
    max_completion_length=8192,
    temperature=1.0,
    # per_device_train_batch_size=4,
    # per_device_train_batch_size=32,
    # per_device_train_batch_size=16,
    per_device_train_batch_size=per_device_train_batch_size,
    # num_generations=(2 * num_gpus - 2 if num_gpus > 1 else 2),
    num_generations=8,
    # gradient_accumulation_steps=int(16 / num_gpus),
    gradient_accumulation_steps=calc_gradient_accumulation_steps(num_gpus, per_device_train_batch_size),
    # gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    save_strategy="steps",
    save_steps=100,
    # save_steps=200,
    save_only_model=True,
    use_vllm=True,
    vllm_device=f"cuda:{num_gpus-1}",
    vllm_gpu_memory_utilization=0.7 if num_gpus > 1 else 0.3,
    logging_steps=1,
    # log_on_each_node=False,
    log_on_each_node=True,
    log_completions=True,
    report_to="wandb",
    apply_chat_template=False,
    )

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    # tokenizer=tokenizer,
    reward_funcs=reward_func,
    train_dataset=train_dataset,
    # eval_dataset=eval_dataset,
)

trainer.train()
