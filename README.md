# `trump-0.0-minus42B` - The Trump LLM

**A really dumb and opinionated LLM — exclusively trained on Donald J. Trump's social media posts[^note].**

Arguably the dumbest Large Language Model ever created.

> [!WARNING]  
> This a work in progress,

[^note]: _Twitter, X and Truth Social posts from 2009 to 2025._

## Why?

This is a self-educational project to learn how to train an LLM from scratch and how to fine-tune pre-trained models.

## Usage

### Getting started

> [!IMPORTANT]  
> **Prerequisites:**
> - **Git LFS** (to download the `data/` directory)
> - **pyenv**
> - **uv**

```sh
git clone https://github.com/ivangabriele/trump-0.0-minus42B.git
cd trump-0.0-minus42B
uv sync
```

### Download Trump's social media posts

> [!WARNING]  
> There are more than 80,000 posts.

```sh
make data
```

### Prepare the training data

_Not ready yet!_

> [!WARNING]  
> There are more than 80,000 posts.

```sh
make prepare
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

_Not ready yet!_

```sh
make tune MODEL=facebook/opt-350m
```

#### Train from scratch

_Not ready yet!_

```sh
make train
```

### Run model (CLI chat)

Starts a CLI chat with the model.

```sh
make run
```
