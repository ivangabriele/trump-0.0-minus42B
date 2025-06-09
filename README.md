# trump-0.0-parameterless

**An LLM exclusively trained on Twitter, X and Truth Social posts by Donald Trump.**

Arguably the dumbest Large Language Model ever created.

## Why

This is a self-educational project to learn how to train an LLM from scratch and how to fine-tune pre-trained models.

## Usage


### Getting started

> [!IMPORTANT] Prerequisites
> - **Git LFS** (to download the `data/` directory)
> - **pyenv**
> - **uv**

```sh
git clone https://github.com/ivangabriele/trump-0.0-parameterless.git
cd trump-0.0-parameterless
uv sync
```


### Download Trump's social media posts

> [!WARNING] There are more than 80,000 posts.

```sh
make data
```

### Prepare the training data

> [!WARNING] There are more than 80,000 posts.

```sh
make scrap
```

### Train model

You have 2 choices here:
- Either fine-tune a pre-trained model, by default `facebook/opt-125m`.
- Or train the model from scratch, which will give you the worst results (but the most fun!).

#### Fine-tune a pre-trained model

Using the default `facebook/opt-125m` model:

```sh
make tune
```

You can also specify a different model:

```sh
make tune MODEL=facebook/opt-350m
```

#### Train from scratch

```sh
make train
```

### Run model

Starts a CLI chat with the model.

```sh
make run
```
