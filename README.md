# trump-0.0-parameterless

**Exclusively trained on Twitter, X and Truth Social posts by Donald Trump. This is arguably the dumbest Large Language Model ever created.**

## Usage

```sh
git clone https://github.com/ivangabriele/trump-0.0-parameterless.git
cd trump-0.0-parameterless
uv sync
```

### Scrap posts

> [!WARNING] There are more than 80,000 posts to scrap, so this may take a while.

```sh
make scrap
```

### Train model

```sh
make train
```

### Run model

Starts a CLI chat with the model.

```sh
make run
```
