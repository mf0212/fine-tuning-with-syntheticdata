## Welcome 💜

Welcome! I've created a platform for fine-tuning Large Language Models (LLMs). I realized that many users lack datasets for fine-tuning LLMs, which is why I built Pluto, a library for synthetic data generation with LLMs. Here's what you can do with it:

- Overcome repetitiveness and make your data highly diverse using topic trees
- Run multiple sampling requests in parallel to speed up data generation
- Use any model provider to generate data

<br>

## Quickstart 🚀

To get started, let's use GPT-4 to generate a dataset of coding questions about numpy. First install the pluto library:


Make sure that you've set your OpenAI API Key as an environment variable:
Then run the following code:

```python
from pluto import EngineArguments, DataEngine, Dataset, TopicTree, TopicTreeArguments

system_prompt = "You are a helpful AI coding assistant. You do not just give high level coding advice, but instead, you respond to coding questions with specific code examples."

tree = TopicTree(
    args=TopicTreeArguments(
        root_prompt="Functionalities of numpy",
        model_system_prompt=system_prompt,
        tree_degree=10,
        tree_depth=2
    )
)

tree.build_tree(model_name="gpt-3.5-turbo-1106")
tree.save("numpy_topictree.jsonl")

engine = DataEngine(
    args=EngineArguments(
        instructions="Please specifically provide training examples with questions about numpy. A training sample should consist of just one question and a response, and not a chat with multiple messages.",
        system_prompt=system_prompt,
        # example_data = Dataset.from_jsonl("example_data.jsonl") | OPTIONAL: comment out this argument to provide examples for the model generating training data

    )
)

dataset = engine.create_data(
    model_name="gpt-4-1106-preview",
    num_steps=20,
    batch_size=5,
    topic_tree=tree
)

dataset.save("output_with_topictree.jsonl")
```

<br>
What happened in this example? 🤔
In the example above, we did the following things:

Generate Topic Tree:
We first used GPT-3.5 to generate a "topic tree" with the root "Functionalities of numpy". A topic tree is simply a tree in which each child of a node needs to be a subtopic of its parent node and allows us to generate a list of aspects that should be covered in our training dataset. This is what paths from root to leaves within a topic tree look like:

```
Functionalities of numpy -> array manipulation -> slicing and indexing
Functionalities of numpy -> matrix operations -> matrix factorization
Functionalities of numpy -> statistical functions -> mean
Functionalities of numpy -> signal processing -> time-frequency analysis
```
<br>
Generate Data from Topic Tree:

After generating our topic tree, we feed it into the create_data function of the DataEngine to ensure that our dataset touches upon a broad range of subjects and is not repetitive. Concretely, in this function, we iterate over all root-to-leaf paths in our topic tree and tell GPT-4 Turbo, which we use to generate our training data, to take the corresponding (sub)topic into account in its generated training sample. The parameter batch_size=5 controls how many OpenAI requests we send simultaneously.

We also provide the option to provide examples of how your dataset should look like to the DataEngine. To do this, simply add example_data=Dataset.from_jsonl('your_data.jsonl') as an argument to DataEngine. Just three or four samples are totally sufficient for your example datasets and help a lot.

<br> Fine-Tune LLMs with your generated Datasets ⚙️
Datasets generated with pluto are saved in a jsonl format:

json
```
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "What's the capital of France?"}, {"role": "assistant", "content": "Paris, as if everyone doesn't know that already."}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "Who wrote 'Romeo and Juliet'?"}, {"role": "assistant", "content": "Oh, just some guy named William Shakespeare. Ever heard of him?"}]}
{"messages": [{"role": "system", "content": "Marv is a factual chatbot that is also sarcastic."}, {"role": "user", "content": "How far is the Moon from Earth?"}, {"role": "assistant", "content": "Around 384,400 kilometers. Give or take a few, like that really matters."}]}
```
You can directly use these dataset files to fine-tune models with various tools and platforms like OpenAI or other training frameworks available.
<br>

## Fine-tuning with Unsloth
### Installation (Local)
+ If you want to finetune on Colab, see this [Colab](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=QmUBVEnvCDJv)

Run on terminal

```
export HF_TOKEN=xxxxxxxxxxxxx
```

+ Conda installation
```
conda create -n unsloth python=3.10 -y
conda activate unsloth
conda install pytorch cudatoolkit torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda install xformers -c xformers

pip install bitsandbytes

pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"

pip install click platformdirs
```

#### Fine-tuning with your own data
```bash
python fine_tuning.py
```
#### Inference

```bash
python inference.py
```