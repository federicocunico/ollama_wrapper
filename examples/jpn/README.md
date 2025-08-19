# Japanese Translation Script

A lightweight CLI utility for translating English (or any language) into **natural Japanese** using an [Ollama](https://ollama.com/) model.  
It uses the model:
```
ollama pull 7shi/gemma-2-jpn-translate:2b-instruct-q8_0
```


## Requirments
Install as main README.md instruction file.

## Usage
### Translate single sentence

```bash
python translate_japanese.py "It's incredible how the sun keeps going up and down every day."
```

### Interactive translation
```bash
python translate_japanese.py
```

### CLI Reference
For reproducibile translations, keep temperature low (e.g., 0.2)

```
positional arguments:
  text                  Text to translate. If omitted, reads from stdin.

optional arguments:
  --url URL             Ollama base URL (default: http://localhost:11434)
  --timeout SECONDS     HTTP timeout in seconds (default: 200)
  --retries N           HTTP retries (default: 5)
  --model NAME          Ollama model name (default: 7shi/gemma-2-jpn-translate:2b-instruct-q8_0)
  --temperature FLOAT   Sampling temperature (default: 0.2)
  --stream              Stream tokens to stdout (if not provided, is False)
  --system PROMPT       Override the system instruction
```
