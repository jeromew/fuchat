# fuchat

## Introduction

Qwen3 LLM inference using the Futhark language + demo chat application

The inference engine has support for KV Cache & Prompt extension.

The chat app has support for
 - user/assistant chat
 - Thinking mode on/off
 - Tool calling of simple futhark entry points

The default model is Qwen3-0.6B and currently needs around 6GB of VRAM


## Usage

You will need
 * a recent nightly release of [Futhark](https://futhark-lang.org/)
 * a python venv with the requirements installed
 
As a first step, compile 
* `futhark {backend} --server qwen-f32.fut -o qwen-f32` for the float32 version (default)
* `futhark {backend} --server qwen-f16.fut -o qwen-f16` for the float16 version
You should probably compile both version so everything's ready for chat tests.

Then you can start the chat with `python chat.py --tools` (or `python chat.py --help` if you want to see the parameters)

and hopefully, after downloading the model weights ( ~1.5GB in float16 format), start chatting with `fuchat` :-)

## Performance

It was developed and tested on an AMD 6700XT with 12GB of VRAM, with the Futhark `hip` backend.

In `f32`mode it reaches around 20-25 token/s and 10 tokens/s in f16. 

As a comparison, on the same card, llama.cpp has an inference of around 150 t/s with the f16 quantized model, and around 110 t/s with the f32 quantized model. 

It is a bit surprising that the f16 version of fuchat is 2 times slower than the f32 version as we could expect a gain from the fact that only half of the memory has to move.

A pure f32 version before implementing KV Cache was running at 2-5 tokens/sec so the Futhark "update in-place" mechanism brings a great performance boost for this type of caching.

Can Futhark reach 100 t/s ? Gotta Go Fast!
It is already impressive that Futhark can reach 25 tokens/s with a one file, typed checked, standalone .fut file !


## Acknowledgements

This is largely inspired by [llaf](https://github.com/BobMcDear/) that created a lightweight GPT2 implementation in Futhark

It wouldn't exist without the help and support of [Troels Henriksen] on subtleties around uniqueness and in-place updates.

## Citations

```bibtex
@software{The_Futhark_Hackers_Futhark,
author = {The Futhark Hackers},
title = {{Futhark}},
url = {https://github.com/diku-dk/futhark}
}
```
```bibtex
@inproceedings{henriksen2017futhark,
  title={Futhark: purely functional GPU-programming with nested parallelism and in-place array updates},
  author={Henriksen, Troels and Serup, Niels GW and Elsman, Martin and Henglein, Fritz and Oancea, Cosmin E},
  booktitle={Proceedings of the 38th ACM SIGPLAN Conference on Programming Language Design and Implementation},
  pages={556*571},
  year={2017}
}
```
```bibtex
@phdthesis{henriksen2017design,
  title={Design and implementation of the Futhark programming language},
  author={Henriksen, Troels},
  year={2017},
  school={University of Copenhagen, Faculty of Science [Department of Computer Science]}
}
```
