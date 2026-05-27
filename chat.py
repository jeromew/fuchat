"""
Runs Qwen3 inference out of a gguf file, with a little chat app
 - thinking mode toggle
 - tool calling
 - kvcache
"""

import argparse
import os
import time
from collections.abc import Mapping
from types import TracebackType
from typing import List, Optional, Type
from pathlib import Path
import requests
import json

import futhark_data
import futhark_server
import numpy as np

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.processors import ByteLevel as ByteLevelPostProcessor
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.normalizers import NFC

from gguf.gguf_reader import GGUFReader

from jinja2 import Environment, BaseLoader
import humanize


class LLM:
    def __init__(self, reader, type, cs, bench) -> None:
        self.server = futhark_server.Server('./qwen-%s' % type, '--cache=qwen-%s.cache' % type)
        self.bench = bench
        self.bench_counter=0
        block_count = reader.get_field('qwen3.block_count').contents()
        self.eos_token_id = reader.get_field("tokenizer.ggml.eos_token_id").contents()
        self.kvcached = 0
        self.tokens = np.array([],  dtype=np.int64)
        total_bytes = 0
        print("-"*80)
        print("Prepare tensors for %s inference inside Futhark server..." % type)
        # map gguf tensors to the `Params` weights
        param_fields = self.server.cmd("fields", "Params")
        param_names = [item.split()[0] for item in param_fields]
        for name in param_names:
            if name in ['token_embd', 'output_norm', 'output']:
                tensor = get_gguf_tensor(reader, "%s.weight" % name)
                if name in ['output']:
                    tensor = np.transpose(tensor)
            else:
                tensor = collect_gguf_tensor(reader, "%s.weight" % name, block_count)
                if name in ['attn_q', 'attn_k', 'attn_v']:
                    # these 3 tensors need to be massaged for the Grouped-Query-Attention
                    # shape: [ blocks, #head or #kv_groups, embedding, head_dimension]
                    tensor = tensor.reshape((*tensor.shape[:-1], -1, 128)).transpose(0,2,1,3)
            if tensor.dtype == np.float16 and type == "f32" :
                tensor = tensor.astype(np.float32)
            print(name, tensor.shape, humanize.naturalsize(tensor.nbytes))
            total_bytes = total_bytes + tensor.nbytes
            self.server.put_value(name, tensor)
        # prepare a `params:Params` variable in the server holding the weights
        self.server.cmd('new', 'params', 'Params', *param_names)
        # free intermediary variables (we keep only `params`)
        for name in param_names:
            self.server.cmd_free(name)

        print("Prepare cache tensors for context size %i" % (cs))
        # qwen3.attention.key_length and qwen3.attention.value_length == 128
        head_dim = reader.get_field('qwen3.attention.key_length').contents()
        block_count = reader.get_field('qwen3.block_count').contents()
        head_count_kv = reader.get_field('qwen3.attention.head_count_kv').contents()
        self.server.put_value('b', np.int64(block_count))
        self.server.put_value('kvh', np.int64(head_count_kv))        
        self.server.put_value('cs', np.int64(cs))        
        self.server.put_value('dh', np.int64(head_dim))
        self.server.cmd_call('init', 'cache', 'b', 'kvh', 'cs', 'dh')
        self.server.cmd_free('b')
        self.server.cmd_free('kvh')
        self.server.cmd_free('cs')
        self.server.cmd_free('dh')

        print("-"*80)
        print("TOTAL VRAM FOR WEIGHTS: ", humanize.naturalsize(total_bytes))

        # Evaluate memory usage of prepared cache
        cache_fields = self.server.cmd("fields", "Cache")
        cache_names = [item.split()[0] for item in cache_fields]
        for name in cache_names:
            self.server.cmd("project", "tensor", "cache", name)
            tensor = self.server.get_value("tensor")
            self.server.cmd_free('tensor')
            print("cache", name, tensor.shape, humanize.naturalsize(tensor.nbytes))
            total_bytes = total_bytes + tensor.nbytes

        print("TOTAL VRAM FOR WEIGHTS+CACHE: ", humanize.naturalsize(total_bytes))

    def __enter__(self) -> 'LLM':
        self.server.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
        ) -> Optional[bool]:
        self.server.cmd_free('params')
        self.server.cmd_free('cache')
        return self.server.__exit(exc_type, exc_value, traceback)

    def shared_prefix_length(self, arr1, arr2):
        min_len = min(len(arr1), len(arr2))
        diff_mask = arr1[:min_len] != arr2[:min_len]
        if not np.any(diff_mask):
            return min_len
        return np.argmax(diff_mask)

    def gen(self, ids: np.ndarray, cnt: int) -> np.ndarray:
        
        # we need to take into account that recent `ids` may have
        # drifted from the kvcache :
        # discard kvcache entries occuring after the drift
        self.kvcached = self.shared_prefix_length(ids, self.tokens)
        self.tokens = self.tokens[:self.kvcached]

        # ids here contains the whole context
        # prompt extension with new tokens
        tokens_in = ids[self.kvcached:]
        self.server.put_value('xsat', np.int64(self.kvcached))
        self.server.put_value('xs', tokens_in)
        self.server.put_value('max_new_tokens', np.int64(cnt))
        self.server.put_value('eos_token_id', np.int64(self.eos_token_id))
        if self.bench:
            self.bench_counter += 1
            if self.bench_counter == self.bench:
                self.server.cmd_store('data.in', 'xsat', 'xs', 'params', 'cache', 'eos_token_id', 'max_new_tokens')
        self.server.cmd_call('gen', 'out', 'xsat', 'xs', 'params', 'cache', 'eos_token_id', 'max_new_tokens')
        self.server.cmd_free('cache')
        self.server.cmd_project('tokens', 'out', '0')
        self.server.cmd_project('cache', 'out', '1')
        tokens_out = self.server.get_value('tokens')
        self.kvcached = self.kvcached + len(tokens_in) + len(tokens_out) - 1
        self.tokens = np.concatenate([self.tokens, tokens_in, tokens_out[:-1]])
        self.server.cmd_free('out')
        self.server.cmd_free('tokens')
        self.server.cmd_free('xsat')
        self.server.cmd_free('xs')
        self.server.cmd_free('eos_token_id')
        self.server.cmd_free('max_new_tokens')

        tokens = np.concatenate([ids, tokens_out])
        return tokens

def get_gguf_tensor(reader, key: str) -> np.ndarray :
    out = None
    for tensor in reader.tensors:
        if tensor.name == key:
            out = tensor.data
    return out

def collect_gguf_tensor(reader, suffix: str, block_count) -> np.ndarray :
    # beware that tensor blocks might not be ordered inside the gguf file
    # we know how many we need to find and collect them in the correct
    # order
    blk = [None] * block_count
    for tensor in reader.tensors:
        if tensor.name.endswith(suffix):
            blk[int(tensor.name.split('.')[1])] = (np.transpose(tensor.data))
    out = np.stack(blk)
    return out

def download_from_huggingface(repo_id, filename, local_dir, revision="main"):
    base_url = "https://huggingface.co"
    url = f"{base_url}/{repo_id}/resolve/{revision}/{filename}"
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    dest_path = os.path.join(local_dir, filename)

    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
    else:
        print(f"Downloading {url} to {dest_path}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return dest_path

def main(
    model: str,
    type: str,
    cs: int,
    cnt: int,
    tools: bool,
    bench: int,
    ) -> None:

    # we always load the f16 version which is the only available
    repo_id = model
    file = model.split('/')[1].replace("-GGUF", "-f16.gguf")
    print("-"*80)
    print("Make sure model is %s/%s available in local dir..." % (repo_id, file))
    download_from_huggingface(repo_id, file, ".")

    print("-"*80)
    print("Start reading gguf file...")
    reader = GGUFReader(file)

    print("-"*80)
    print("Prepare tokenizer from gguf file...")

    # the commented single line AutoTokenizer call is more generic (handle the tokenizer for many models) but is very slow
    # to setup (around 30s)
    # (if uncommented it will also need some changes as it AutoTokenizer has some API differences)
    #tokenizer = AutoTokenizer.from_pretrained(".", gguf_file=file)

    # configure comptabible Tokenizer directly
    ggml_tokens = reader.get_field("tokenizer.ggml.tokens").contents()
    ggml_token_type = reader.get_field("tokenizer.ggml.token_type").contents()
    ggml_merges = reader.get_field("tokenizer.ggml.merges").contents()
    tokenizer = Tokenizer(BPE(
        vocab={token: idx for idx, token in enumerate(ggml_tokens)},
        merges=[tuple(merge.split()) for merge in ggml_merges],
    ))
    tokenizer.normalizer = NFC()
    tokenizer.add_special_tokens([ggml_tokens[idx] for idx in range(len(ggml_tokens)) if ggml_token_type[idx] in (3,4)])
    tokenizer.decoder = ByteLevelDecoder()
    regex_pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
    tokenizer.pre_tokenizer = Sequence([
        Split(pattern=regex_pattern, behavior="isolated", invert=False),
        ByteLevelPreTokenizer(add_prefix_space=False, trim_offsets=False, use_regex=False)
    ])
    tokenizer.post_processor = ByteLevelPostProcessor(
        add_prefix_space=False,
        trim_offsets=False,
        use_regex=False
    )

    # prepare the chat jinja2 template
    # note that this template is not able to create tokens in an idempotent way
    # Each role:user messages deletes the previous reasoning traces
    # This needs to be taken into account for optimizing inference vs kvcache drift
    chat_template = reader.get_field("tokenizer.chat_template").contents()
    rtemplate = Environment(loader=BaseLoader).from_string(chat_template)
    eos_token_id = reader.get_field("tokenizer.ggml.eos_token_id").contents()

    # start the futhark server    
    llm = LLM(reader, type, cs, bench)

    messages = []#[{ "role": "system", "content": "Make sure to give your answer and then add Gotta Go Fast! on a new line" }] #[] for no system message
    enable_thinking = False
    context_size = 0
    show_context = False

    if len(messages):
        print("System Prompt: %s" % messages[0]['content'])

    #tools = False
    available_tools = { name.split("_")[1]: " ".join(name.split("_")[2:]) for name in llm.server.cmd("entry_points") if name.startswith("tool_") }
    if tools:
        tools = []
        print("-"*80)
        print("Registering tools for tool calling")
        for key, value in available_tools.items():
            print(" - %s: %s" % (key, value))
            tools.append({
                "type": "function",
                "function": {
                    "name": key,
                    "description": "calculate the %s" % value,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n": {
                                "type": "number",
                                "description": "",
                            },
                        }
                    }
                }
            })
    print("-"*80)
    print("Special in-chat commands: quit/clear/think/nothink/show/hide")
    print("What can I do for you today ?")
    
    # start chat loop
    looping = True
    while looping:
        if (llm.kvcached == cs):
            print("\n<=== Context of size %i is Full ===>. Use `clear` to wipe it out" % cs)
        if len(messages) == 0 or messages[-1]["role"] != "tool":
            user_message = input("(ctx: %i/%i)> " % (context_size, cs))
            if user_message == "quit":
                print("bye!")
                break
            if user_message == "clear":
                print("context cleared")
                messages = [m for m in messages if m["role"] == "system" ]
                context_size = 0
                llm.kvcached = 0
                llm.tokens = np.array([],  dtype=np.int64)
                continue
            if user_message == "show":
                print("context is displayed when sending message")
                show_context = True
                if len(messages):
                    print("-"*27, " current messages context ", "-"*27)
                    print(rtemplate.render({ "tools": tools, "messages": messages }))
                    print("-"*80)
                continue
            if user_message == "hide":
                print("context is not displayed when sending message, only the assistant's answer")
                show_context = False
                continue
            if user_message == "think":
                enable_thinking = True
                print("thinking mode enabled")
                continue
            if user_message == "nothink":
                enable_thinking = False
                print("thinking mode disabled")
                continue
            messages.append({ "role": "user", "content": user_message })

        if (llm.kvcached == cs):
            continue

        in_text = rtemplate.render({ "tools": tools, "messages": messages, "add_generation_prompt": True, "enable_thinking": enable_thinking })

        if show_context:
            print("-"*33, " sent context ", "-"*33)
            print(in_text)
            print("-"*80)

        start = time.time()
        count_tokens = 0
        ids = np.array(tokenizer.encode(in_text).ids, dtype=np.int64)
        assistant_message_ids = np.array([],  dtype=np.int64)
        reasoning_capture = np.array([],  dtype=np.int64)
        reasoning_registering = False
        reasoning_content = None
        tool_call_capture = np.array([],  dtype=np.int64)
        tool_call_registering = False
        tool_calls = []
        while ids[-1] != eos_token_id and len(ids) <= cs:
            out = llm.gen(ids, cnt)
            limit = len(out)
            if (out[-1] == eos_token_id):
                limit = -1
            streaming = np.array([],  dtype=np.int64)
            for id in out[len(ids):limit]:
                count_tokens = count_tokens + 1
                if id == tokenizer.token_to_id("<think>"):
                    reasoning_registering = True
                    continue
                if id == tokenizer.token_to_id("<tool_call>"):
                    tool_call_registering = True
                    continue
                if id == tokenizer.token_to_id("</think>"):
                    reasoning_registering = False
                    reasoning_content = tokenizer.decode(reasoning_capture.tolist(), skip_special_tokens=False).strip()
                    reasoning_capture = np.array([],  dtype=np.int64)
                    continue
                if id == tokenizer.token_to_id("</tool_call>"):
                    tool_call_registering = False
                    tool_calls.append(json.loads(tokenizer.decode(tool_call_capture.tolist(), skip_special_tokens=False).strip()))
                    tool_call_capture = np.array([],  dtype=np.int64)
                    continue
                if reasoning_registering:
                    reasoning_capture = np.append(reasoning_capture, id)
                elif tool_call_registering:
                    tool_call_capture = np.append(tool_call_capture, id)
                else:
                    streaming = np.append(streaming, id)
            if show_context:
                print(tokenizer.decode(out[len(ids):limit].tolist(), skip_special_tokens=False), end='', flush=True)
            else:
                print(tokenizer.decode(streaming.tolist(), skip_special_tokens=False), end='', flush=True)
            assistant_message_ids = np.append(assistant_message_ids, streaming)
            if limit == -1:
                if reasoning_registering or tool_call_registering:
                    print("\n !!!anomaly: <think> tag or <tool_call> tag not closed by the model despite eos_token_id")
                    if reasoning_registering:
                        reasoning_registering = False
                        reasoning_content = tokenizer.decode(reasoning_capture.tolist(), skip_special_tokens=False).strip()
                        reasoning_capture = np.array([],  dtype=np.int64)
                messages.append({ 
                    "role": "assistant",
                    "content": tokenizer.decode(assistant_message_ids.tolist(), skip_special_tokens=False).strip(),
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls
                })
                for tool_call in tool_calls:
                    if tool_call["name"] in available_tools:
                      llm.server.put_value('n', np.int64(tool_call["arguments"]["n"]))
                      llm.server.cmd_call('tool_%s_%s' % (tool_call["name"], available_tools[tool_call["name"]].replace(" ", "_")), 'out', 'n')
                      out = llm.server.get_value('out')
                      messages.append({ 
                          "role": "tool",
                          "content": str(out),
                      })
                      llm.server.cmd_free('out')
                      llm.server.cmd_free('n')
                    else:
                        print("Error: wrong tool name %s" % tool_call["name"])
                context_size = len(ids) + len(assistant_message_ids)
                end = time.time()
                print(f'\nt/s: {count_tokens/(end-start):.3f} s')
                break
            else:
                ids = out



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs GPT-2 inference using llm.')
    parser.add_argument('--model',
                        type=str,
                        default='ggml-org/Qwen3-0.6B-GGUF',
                        help='Qwen3 variant to use.',
                        choices=['ggml-org/Qwen3-0.6B-GGUF', 'ggml-org/Qwen3-1.7B-GGUF'])
    parser.add_argument('--type',
                        type=str,
                        default='f32',
                        help='tensor types in futhark (default f32)',
                        choices=['f32', 'f16'])
    parser.add_argument('--cs',
                        type=int,
                        default=8192,
                        help='maximum context size (default 8192)')
    parser.add_argument('--cnt',
                        type=int,
                        default=5,
                        help='a maximum of cnt tokens is generated/streamed at every step of the inference (default 5)')
    parser.add_argument("--tools",
                        action='store_true',
                        default=False,
                        help='activate tool discovery & tool calling')
    parser.add_argument('--bench',
                        type=int,
                        default=0,
                        help='the application runs in benchmark mode. cf https://futhark-lang.org/blog/2026-05-22-benchmarking-a-real-futhark-application.html for more information')

    args = parser.parse_args()
    main(args.model, args.type, args.cs, args.cnt, args.tools, args.bench)
