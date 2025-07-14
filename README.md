Experiments with using a steering direction that converts from "this case" to "THIS CASE".

Usage:

`uv run find_upper_dir.py`
`uv run main.py`

I experimented a bit with this, and it seems the idea has some merit to it.

It looked plausible for a simple model like `EleutherAI/pythia-410m` but foundered on `Qwen/Qwen2-7B-Instruct`

But I took a very simplistic pairwise per-token approach. This fails for fairly common cases:

```
tokenizer.convert_ids_to_tokens(tokenizer.encode(" is the correct answer"))
['Ġis', 'Ġthe', 'Ġcorrect', 'Ġanswer']
tokenizer.convert_ids_to_tokens(tokenizer.encode(" IS THE CORRECT ANSWER"))
['ĠIS', 'ĠTHE', 'ĠCOR', 'RECT', 'ĠANSW', 'ER']
```

I think there are sitll some directions this could go, but am time boxing:
* we might still get good value simply out of only using the tokens MUST, SHOULD, etc
* we could find a direction in a later layer that actually corresponds to something being suddenly allcaps (as opposed to just being part of an all caps sentence)
* Test why " IS THE correct answer" isn't sufficient hinting.
* Find another model with a less crazy tokenization
* Isn't it strange that there isn't a generalized uppercase dimension. It seems such an obvious thing to learn. How else could it be encoded?
* Suggestion from Ed: Use "." -> "!" as the salience map instead. Probably simpler all round.