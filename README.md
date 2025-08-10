# Ollama wrapper

## Requirements:

- Python 3.10+
- Ollama

## Installation

In a terminal, run the following command:

```bash
cd <project_directory>
pip install poetry
poetry install
pip install .
```

Check installation:
```bash
python -c "import ollama_wrapper"
```

If the output is empty, the installation was successful.


## Examples

### Translation

> NOTE: Ensure the book is saved as .txt with `utf-16` encoding. Use any spacer string you like.

Example to translate any book in Italian with spacer for chapters as `$$$$`, using the `aya-expanse:latest` (the 8B version).

Download required Ollama models with:

```bash
ollama pull aya-expanse:latest
```

Then execute as:

```bash
python examples/translate.py --input "yourbook.txt" --spacer "$$$$" --language "Italian" --model "aya-expanse:latest"
```

It will take time. Results will be saved in the directory `translated/<bookname>`.
