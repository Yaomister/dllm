from datasets import load_dataset

if __name__ == "__main__":
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    print(dataset.column_names)

    
