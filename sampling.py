from transformers import AutoModel
from datasets import load_dataset
from schedulers import CosScheduler, LinearScheduler, ConstantScheduler, InverseScheduler

schedulers = [ConstantScheduler, LinearScheduler, CosScheduler, InverseScheduler]

def load_database():
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    return dataset


def load_model():
    model = AutoModel.from_pretrained(pretrained_model_name_or_path="GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    return model


def run(model, database):
    return

if __name__ == "__main__":
    database = load_database()
    model = load_model()

    run(model, database)