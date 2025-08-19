# Translation to European languages

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
