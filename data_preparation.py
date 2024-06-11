import json
from pluto import EngineArguments, DataEngine, Dataset, TopicTree, TopicTreeArguments
from datasets import Dataset as HFDataset, DatasetDict
from huggingface_hub import HfApi

# Define system prompt
system_prompt = "You are a helpful AI math assistant. You help students with their questions and give anwers for them. You do not just give high level mathing advice, but instead, you tend to respond to math questions with specific math examples. Give answer in Vietnamese"

# Create and build the topic tree
tree = TopicTree(
    args=TopicTreeArguments(
        root_prompt="Math in secondary School",
        model_system_prompt=system_prompt,
        tree_degree=10,
        tree_depth=2
    )
)

tree.build_tree(model_name="gpt-3.5-turbo")
tree.save("numpy_topictree.jsonl")

# Create the data engine
engine = DataEngine(
    args=EngineArguments(
        instructions="Please specifically provide training examples with questions about math. A training sample should consist of just one question and a response, and not a chat with multiple messages.",
        system_prompt=system_prompt,
    )
)

# Generate the dataset
dataset = engine.create_data(
    model_name="gpt-3.5-turbo",
    num_steps=20,
    batch_size=5,
    topic_tree=tree
)

# Save the dataset locally
dataset.save("math.jsonl")

# Load the saved dataset
with open("math.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
    data = [json.loads(line) for line in lines]

# Convert to Hugging Face Dataset format
hf_dataset = HFDataset.from_list(data)

# Create a dataset dictionary to prepare for uploading
dataset_dict = DatasetDict({"train": hf_dataset})

# Save the dataset to a Hugging Face repository
dataset_dict.push_to_hub("mf212/math-dataset")

# Authenticate and upload the dataset
api = HfApi()
api.upload_file(
    path_or_fileobj="math.jsonl",
    path_in_repo="math.jsonl",
    repo_id="mf212/math-dataset",
    repo_type="dataset"
)
