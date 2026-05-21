-- Qwen3 LLM inference in Futhark.
-- All the relevant code is contained in this file,
-- and no imports are necessary besides the prelude.
-- There are two points of entry:
-- init: Initializes the model KV cache and some pre-calculated weights
-- gen: Inference with prompt extension & in-place kvcache update

-- The model's architecture is fully specified by the following hyperparameters:
-- Example values are given for Qwen3-0.6B
-- [v]: Vocabulary size. The number of possible i64 tokens (ex: 151936)
-- [d]: Embedding dimensionality. Each token is mapped to a [d]-dim vector (ex: 1024)
-- [b]: n_layers - Number of transformer blocks. (ex: 28)
-- [f]: Dimensionality of the FeedForward hidden layer (ex: 3072)
-- [h]: Number of heads of the query (ex: 16)
-- [kvh]: Number of kv groups in "Grouped Query Attention" (ex: 8)
-- [dh]: All Q/K/V heads have this dimensionality (ex: 128)

-- Note that a leading [b] axis means the ith entry belongs to the ith transformer block.
type Params [b][v][d][f][h][kvh][dh] = {
    -- Token Embeddings
    token_embd: [v][d]f16,
    output_norm: [d]f32,
    output: [d][v]f16,
    attn_norm: [b][d]f32,
    ffn_norm: [b][d]f32,
    ffn_gate: [b][d][f]f16,
    ffn_up: [b][d][f]f16,
    ffn_down: [b][f][d]f16,
    attn_q: [b][h][d][dh]f16,
    attn_k: [b][kvh][d][dh]f16,  -- Attention K weights
    attn_v: [b][kvh][d][dh]f16, -- Attention V weights
    attn_output: [b][h*dh][d]f16, -- Attention projection weights
    q_norm: [b][dh]f32, -- Normalization weights for Q
    k_norm: [b][dh]f32, -- Normalization weights for K
}

-- mask: Causal self-attention mask.
type Cache [b][kvh][cs][dh] = {
    mask: [cs][cs]f16,
    cos: [cs][dh]f16,
    sin: [cs][dh]f16,
    kcache: [b][kvh][cs][dh]f16,
    vcache: [b][kvh][cs][dh]f16,
}


def matmul [n][m][p] (A: [n][m]f16) (B: [m][p]f16): [n][p]f16 =
    let dot_prod (a: [m]f16) (b: [m]f16): f16 =
        reduce (+) 0.0 (map2 (*) a b)
    in map (\a -> map (dot_prod a) (transpose B)) A

def softmax [n] (a: [n]f16): [n]f16 = -- Operates over vectors, later mapped over matrices.
    let shifted = map ((+) (-(reduce f16.max a[0] a))) a -- Subtracts max for stability.
    let es = map f16.exp shifted
    let sum = reduce (+) 0.0 es
    in map (\e -> e / sum) es

def argmax [n] (a: [n]f16): i64 = -- Operates over vectors, later mapped over matrices.
    let update ((mx_v, mx_i): (f16, i64)) ((curr_v, curr_i): (f16, i64)) =
        if mx_v < curr_v then (curr_v, curr_i) else (mx_v, mx_i)
    let (_, mx_i) = reduce_comm update (a[0], 0) (zip a (iota n)) -- reduce_comm is better optimized.
    in mx_i

-- FeedForward
def ff [l][d][f] (xs: [l][d]f16) (ffn_gate: [d][f]f16) (ffn_up: [d][f]f16) (ffn_down: [f][d]f16): [l][d]f16 =
    let xs_ffn_gate_silu = map (\d1 -> map (\d2 -> d2 / (1 + f16.exp (f16.neg d2))) d1) (matmul xs ffn_gate)
    let xs_ffn_up = matmul xs ffn_up
    let x = map2 (map2 (*)) xs_ffn_gate_silu xs_ffn_up
    in matmul x ffn_down

-- RMSNorm
def rms_norm [l][d] (xs: [l][d]f16) (gamma: [d]f32): [l][d]f16 =
    let norm_row (x: [d]f16): [d]f16 =
        let var = (reduce (+) 0.0 (map (\xi -> (f32.f16 xi) * (f32.f16 xi)) x)) / f32.i64 d
        let rms = f32.sqrt (var + 1e-6)
        in map2 (\xi gi -> f16.f32 (gi * (f32.f16 xi) / rms)) x gamma
    in map norm_row xs

-- apply RoPE Rotary Positional Embedding to give tokens a sense of their relative positioning
def apply_rope [T][dh] (xs: [T][dh]f16) (cos: [T][dh]f16) (sin: [T][dh]f16): [T][dh]f16 =
    let x1 = map (take (dh/2)) xs
    let x2 = map (drop (dh/2)) xs
    let rotated = map2 (\row1 row2 -> (map f16.neg row2) ++ row1) x1 x2 :> [T][dh]f16
    in map4 (map4 (\x c xr s -> x * c + xr * s)) xs cos rotated sin

-- evaluate grouped query attention with kvcache
def gqa [b][cs][T][d][h][kvh][dh] (bidx: i64) (xsat: i64) (xs: [T][d]f16)
                         (attn_q: [h][d][dh]f16)
                         (attn_k: [kvh][d][dh]f16)
                         (attn_v: [kvh][d][dh]f16)
			 (attn_output: [h*dh][d]f16)
			 (q_norm: [dh]f32)
			 (k_norm: [dh]f32)
			 (cache: *Cache [b][kvh][cs][dh])
                         : ([T][d]f16, *Cache [b][kvh][cs][dh]) =
    let s = f16.sqrt (1 / f16.i64 dh) -- Scale factor.
    let qs = map (\head -> apply_rope (rms_norm (matmul xs head) q_norm) (cache.cos[xsat:xsat+T, :] :> [T][dh]f16) (cache.sin[xsat:xsat+T, :]  :> [T][dh]f16)) attn_q
    let kvgroup_ks = map (\head -> apply_rope (rms_norm (matmul xs head) k_norm) (cache.cos[xsat:xsat+T, :] :> [T][dh]f16) (cache.sin[xsat:xsat+T, :]  :> [T][dh]f16)) attn_k
    let kvgroup_vs = map (\head -> matmul xs head) attn_v
    -- note that we cache K/V with RoPE. This works as long as we do not use the kvcache as a ring buffer where the array indices may not reflect tokens ordering
    let kvcache = cache with kcache[bidx, :, xsat:xsat+T] = kvgroup_ks with vcache[bidx, :, xsat:xsat+T] = kvgroup_vs    
    -- expand kvgroup to meet the number of Q heads (Grouped _Query_ Attention)
    let ks = flatten (map (replicate (h/kvh)) (kvcache.kcache[bidx, :, :xsat+T])) :> [h][xsat+T][dh]f16
    let vs = flatten (map (replicate (h/kvh)) (kvcache.vcache[bidx, :, :xsat+T])) :> [h][xsat+T][dh]f16
    -- calculate attention with causal mask
    let raw_att = map2 (\q k -> matmul q (transpose k)) qs ks |> map (map (map (\a -> a * s)))
    let att = map (\head -> map2 (map2 (+)) head (kvcache.mask[xsat:xsat+T, :xsat+T] :> [T][xsat+T]f16) |> map softmax) raw_att
    let conts = map2 matmul att vs |> transpose |> map flatten
    in (matmul conts attn_output, kvcache)

-- Passes the input through the ith block.
let block [T][cs][b][v][d][f][h][kvh][dh] (xsat: i64) (xs: [T][d]f16) (i: i64) (ps: Params [b][v][d][f][h][kvh][dh]) (cache: *Cache [b][kvh][cs][dh]): ([T][d]f16, *Cache [b][kvh][cs][dh]) = -- Passes the input through the ith block.
    let ln1 = rms_norm xs ps.attn_norm[i]
    let attn = gqa i xsat ln1 ps.attn_q[i] ps.attn_k[i] ps.attn_v[i] ps.attn_output[i] ps.q_norm[i] ps.k_norm[i] cache
    let y1 = map2 (map2 (+)) xs attn.0
    let ln2 = rms_norm y1 ps.ffn_norm[i]
    in (map2 (map2 (+)) y1 (ff ln2 ps.ffn_gate[i] ps.ffn_up[i] ps.ffn_down[i]), attn.1)

-- transform T tokens positioned at xsat and returns for each token a probability of its next token over the vocabulary
def transformer [T][cs][b][v][d][f][h][kvh][dh] (xsat: i64) (xs: [T]i64) (ps: Params [b][v][d][f][h][kvh][dh]) (cache: *Cache [b][kvh][cs][dh]): ([T][v]f16, *Cache [b][kvh][cs][dh]) =
    -- embed the tokens
    let xs = (map (\id -> ps.token_embd[i64.min (v-1) id]) xs)
    -- sequential evaluation of the layer blocks
    let ys = loop (xs, cache) = (xs, cache) for i < b do block xsat xs i ps cache
    -- normalize and project on vocabulary
    in (matmul (rms_norm ys.0 ps.output_norm) ps.output, ys.1)

-- absorb prompt extension and generate 1 new token
def gen_token [T][cs][b][v][d][f][h][kvh][dh] (xsat: i64) (xs: [T]i64) (max_new_tokens: i64) (i: i64) (new_tokens: *[max_new_tokens]i64) (ps: Params [b][v][d][f][h][kvh][dh]) (cache: *Cache [b][kvh][cs][dh]): (i64, *Cache [b][kvh][cs][dh], i64, []i64, *[max_new_tokens]i64) =
    let zs = transformer xsat xs ps cache
    -- choose the next token as the token with the highest probability
    -- this is where tok_k, temperature, .. could be implemented
    let new_token = last zs.0 |> argmax
    in (i+1,  zs.1, xsat+T, [new_token], new_tokens with [i] = new_token)

-- `gen` autoregressively infer tokens with prompt extension and kvcache
-- xsat, xs is the prompt extension xs inserted at position xsat in the context
-- inference stops upon generating eos_token_id
-- inference also stops when a maximum of max_new_tokens has been generated
-- or when the context is full
--
entry gen [cs][b][v][d][f][h][kvh][dh] (xsat: i64) (xs: []i64) (ps: Params [b][v][d][f][h][kvh][dh]) (cache: *Cache [b][kvh][cs][dh]) (eos_token_id: i64) (max_new_tokens: i64) : []i64 =
    let res = loop (i, cache, xsat, xs: []i64, new_tokens) = (0, cache, xsat, xs, replicate max_new_tokens 0)
        while (i < max_new_tokens && (xsat + i) < cs && new_tokens[i64.max 0 i-1] != eos_token_id)
            do gen_token xsat xs max_new_tokens i new_tokens ps cache
    in take res.0 res.4

-- `init` is called during server instanciation to pre-build some arrays
-- mask: causal mask used for attention
-- cos/sin: used for RoPE - Rotary Positional Embedding
-- kcache/vcache: kvcache
--
entry init (b: i64) (kvh: i64) (cs: i64) (dh: i64): Cache [b][kvh][cs][dh] = {
    mask=tabulate_2d cs cs (\i j -> if j > i then -f16.inf else 0.0),
    cos=map2 (\i r -> map (\c -> f16.f32(f32.cos(c*(f32.i64 i)))) r) (iota cs) (replicate cs (flatten (replicate 2 (map (\i -> (1 / (1000000f32 ** ((f32.i64 i) * 2 / f32.i64 dh)))) (iota 64))))) :> [cs][dh]f16,
    sin=map2 (\i r -> map (\c -> f16.f32(f32.sin(c*(f32.i64 i)))) r) (iota cs) (replicate cs (flatten (replicate 2 (map (\i -> (1 / (1000000f32 ** ((f32.i64 i) * 2 / f32.i64 dh)))) (iota 64))))) :> [cs][dh]f16,
    kcache=replicate b (replicate kvh (replicate cs (replicate dh 0))),
    vcache=replicate b (replicate kvh (replicate cs (replicate dh 0))),
}

-- demo with tool calling
-- entry points beginning with tool_ are automatically registered as tool_name_description
-- the current implementation only handles (n: i64): i64

entry tool_sumN1_sum_of_first_n_integers (n: i64): i64 =
    reduce (+) 0 (iota (n+1))

entry tool_sumN2_sum_of_squares_of_first_n_integers (n: i64): i64 =
    reduce (+) 0 (map (**2) (iota (n+1)))