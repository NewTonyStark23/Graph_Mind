export interface DeepMethod {
  id: string;
  name: string;
  category: 'architecture' | 'weights' | 'behavior' | 'merging' | 'interpretability';
  description: string;
  technique: string;
  tools: string[];
  codeExample: string;
  outputData: string;
  relevance: string;
  difficulty: 'beginner' | 'intermediate' | 'advanced' | 'expert';
}

export const deepMethods: DeepMethod[] = [
  {
    id: 'weight-subspace-svd',
    name: 'Weight Subspace Decomposition (SVD)',
    category: 'weights',
    description: 'Decompose weight matrices into U, S, V^T via SVD to reveal principal components of learned knowledge.',
    technique: 'Truncated SVD on each linear layer',
    tools: ['numpy.linalg.svd', 'torch.linalg.svd'],
    codeExample: `import numpy as np\nfrom safetensors import safe_open\n\nwith safe_open("model.safetensors", framework="numpy") as f:\n    for key in f.keys():\n        if "weight" in key:\n            W = f.get_tensor(key)\n            U, S, Vt = np.linalg.svd(W, full_matrices=False)\n            energy = np.cumsum(S**2) / np.sum(S**2)\n            print(f"{key}: rank={len(S)}, top-100 energy={energy[99]:.3f}")`,
    outputData: 'Per-layer singular value spectra, principal components',
    relevance: 'Models with similar SVD spectra can merge safely',
    difficulty: 'advanced'
  },
  {
    id: 'activation-clustering',
    name: 'Activation Pattern Clustering',
    category: 'behavior',
    description: 'Collect intermediate activations and cluster to discover semantic structure.',
    technique: 'Forward hooks + PCA + HDBSCAN clustering',
    tools: ['transformers hooks', 'sklearn.cluster.HDBSCAN'],
    codeExample: `from transformers import AutoModelForCausalLM\nimport torch\n\nmodel = AutoModelForCausalLM.from_pretrained("model")\nactivations = {}\n\ndef hook_fn(name):\n    def fn(module, input, output):\n        activations[name] = output[0].detach().cpu()\n    return fn\n\nfor name, layer in model.model.layers.named_children():\n    layer.register_forward_hook(hook_fn(f"layer_{name}"))`,
    outputData: 'Per-layer activation matrices, cluster assignments',
    relevance: 'Shared activation clusters indicate compatible representations',
    difficulty: 'expert'
  },
  {
    id: 'attention-head-roles',
    name: 'Attention Head Role Classification',
    category: 'interpretability',
    description: 'Classify what each attention head does: positional, induction, semantic, etc.',
    technique: 'Analyze attention patterns and position bias',
    tools: ['transformer-lens', 'bertviz'],
    codeExample: `def classify_head(attention_pattern):\n    heads, seq, _ = attention_pattern.shape\n    roles = []\n    for h in range(heads):\n        pattern = attention_pattern[h].mean(dim=0)\n        pos_entropy = -np.sum(pattern * np.log(pattern + 1e-10))\n        is_positional = pos_entropy < 2.0\n        role = "positional" if is_positional else "semantic"\n        roles.append(role)\n    return roles`,
    outputData: 'Per-head role labels, attention entropy scores',
    relevance: 'CRITICAL: Must align heads by role, not position, when merging',
    difficulty: 'expert'
  },
  {
    id: 'tokenizer-alignment',
    name: 'Tokenizer Vocabulary Alignment',
    category: 'merging',
    description: 'Map tokens between models to align embedding layers before merging.',
    technique: 'String matching + embedding similarity',
    tools: ['tokenizers', 'scipy.optimize.linear_sum_assignment'],
    codeExample: `from tokenizers import Tokenizer\nimport numpy as np\n\ntok_a = Tokenizer.from_file("model_a/tokenizer.json")\ntok_b = Tokenizer.from_file("model_b/tokenizer.json")\n\nvocab_a = {tok_a.id_to_token(i): i for i in range(tok_a.get_vocab_size())}\nvocab_b = {tok_b.id_to_token(i): i for i in range(tok_b.get_vocab_size())}\n\nshared = set(vocab_a.keys()) & set(vocab_b.keys())\nprint(f"Shared tokens: {len(shared)}")`,
    outputData: 'Token mapping table, alignment projection matrix',
    relevance: 'THE #1 cause of merge crashes. Always align tokenizers first!',
    difficulty: 'intermediate'
  },
  {
    id: 'residual-stream-analysis',
    name: 'Residual Stream Information Flow',
    category: 'interpretability',
    description: 'Track information flow through residual stream to identify critical layers.',
    technique: 'Measure norm growth and layer contributions',
    tools: ['torch', 'custom hooks'],
    codeExample: `def analyze_residual_stream(model, tokens):\n    residual_norms = []\n    layer_contributions = []\n    \n    with torch.no_grad():\n        x = model.embed_tokens(tokens)\n        residual_norms.append(x.norm(dim=-1).mean().item())\n        \n        prev_x = x.clone()\n        for i, layer in enumerate(model.layers):\n            x = layer(x)[0]\n            delta = (x - prev_x).norm(dim=-1).mean()\n            layer_contributions.append(delta.item())\n            residual_norms.append(x.norm(dim=-1).mean().item())\n            prev_x = x.clone()\n    \n    return layer_contributions`,
    outputData: 'Per-layer contribution scores, critical layer identification',
    relevance: 'Weight critical layers more carefully during merge',
    difficulty: 'advanced'
  },
  {
    id: 'moe-expert-utilization',
    name: 'MoE Expert Utilization Mapping',
    category: 'behavior',
    description: 'Map which experts activate for which inputs to build specialization profiles.',
    technique: 'Route dataset through model, log expert selections',
    tools: ['transformers', 'custom MoE hooks'],
    codeExample: `expert_usage = {i: [] for i in range(num_experts)}\n\ndef moe_hook(module, input, output):\n    router_logits = output[1]\n    topk_indices = router_logits.topk(k=top_k, dim=-1).indices\n    for batch_idx in range(topk_indices.shape[0]):\n        for token_idx in range(topk_indices.shape[1]):\n            for expert_idx in topk_indices[batch_idx, token_idx]:\n                expert_usage[expert_idx.item()].append(current_prompt_category)\n\nfor expert_id, categories in expert_usage.items():\n    profile = Counter(categories).most_common(5)\n    print(f"Expert {expert_id}: {profile}")`,
    outputData: 'Expert specialization profiles, utilization frequency',
    relevance: 'Match experts by specialization when merging MoE models',
    difficulty: 'expert'
  },
  {
    id: 'weight-sparsity-pattern',
    name: 'Weight Sparsity & Magnitude Patterns',
    category: 'weights',
    description: 'Analyze weight magnitude distributions and sparsity patterns per layer.',
    technique: 'Compute histograms, identify outliers, measure effective rank',
    tools: ['numpy', 'scipy.stats'],
    codeExample: `from safetensors import safe_open\nimport numpy as np\nfrom scipy import stats\n\nwith safe_open("model.safetensors", framework="numpy") as f:\n    for key in f.keys():\n        if "weight" in key:\n            W = f.get_tensor(key).flatten()\n            mean, std = np.mean(W), np.std(W)\n            sparsity = np.mean(np.abs(W) < 1e-4)\n            outliers = np.mean(np.abs(W - mean) > 3 * std)\n            skewness = stats.skew(W)\n            kurtosis = stats.kurtosis(W)\n            print(f"{key}: sparsity={sparsity:.3f}, outliers={outliers:.3f}")`,
    outputData: 'Per-layer sparsity maps, outlier locations, distribution shapes',
    relevance: 'Dense + sparse layers create corruption when merged',
    difficulty: 'intermediate'
  },
  {
    id: 'logit-lens',
    name: 'Logit Lens — Layer Output Decoding',
    category: 'interpretability',
    description: 'Apply unembedding to intermediate layers to see what each layer predicts.',
    technique: 'Hidden states × lm_head → predictions per layer',
    tools: ['transformer-lens', 'transformers hooks'],
    codeExample: `def logit_lens(model, tokens, target_position):\n    results = []\n    with torch.no_grad():\n        x = model.embed_tokens(tokens)\n        for i, layer in enumerate(model.layers):\n            x = layer(x)[0]\n            logits = model.lm_head(x)\n            probs = torch.softmax(logits[0, target_position], dim=-1)\n            top_tokens = probs.topk(10)\n            results.append({\n                "layer": i,\n                "top_predictions": [\n                    (tokenizer.decode([t.item()]), p.item())\n                    for t, p in zip(top_tokens.indices, top_tokens.values)\n                ]\n            })\n    return results`,
    outputData: 'Per-layer token predictions, knowledge emergence curves',
    relevance: 'Shows WHERE knowledge lives in the network',
    difficulty: 'advanced'
  },
  {
    id: 'embedding-geometry',
    name: 'Embedding Space Geometry Analysis',
    category: 'architecture',
    description: 'Analyze geometric structure of embedding space: distances, directions, isotropy.',
    technique: 'Pairwise cosine distances, relational consistency, isotropy measurement',
    tools: ['numpy', 'scipy.spatial'],
    codeExample: `def analyze_embedding_geometry(embeddings, vocab, concepts):\n    vecs = {word: embeddings[vocab[word]] for word in concepts}\n    \n    # Check relational consistency\n    relation = vecs["king"] - vecs["man"] + vecs["woman"]\n    nearest = cosine_similarity([relation], embeddings)[0].argmax()\n    \n    # Measure isotropy\n    cov = np.cov(embeddings.T)\n    eigenvalues = np.linalg.eigvalsh(cov)\n    isotropy = eigenvalues.min() / eigenvalues.max()\n    \n    return {"isotropy": isotropy}`,
    outputData: 'Isotropy scores, relational consistency metrics',
    relevance: 'Mismatched isotropy creates broken geometry after merge',
    difficulty: 'advanced'
  },
  {
    id: 'merge-compatibility-score',
    name: 'Pre-Merge Compatibility Scoring',
    category: 'merging',
    description: 'Compute comprehensive compatibility score before attempting merge.',
    technique: 'Weighted combination of architecture, vocab, spectral, activation, loss similarity',
    tools: ['All above methods combined'],
    codeExample: `def merge_compatibility_score(model_a_path, model_b_path):\n    scores = {}\n    \n    # 1. Architecture match\n    config_a = load_config(model_a_path)\n    config_b = load_config(model_b_path)\n    scores["arch"] = 1.0 if configs_compatible(config_a, config_b) else 0.0\n    \n    # 2. Vocabulary overlap\n    vocab_a = load_vocab(model_a_path)\n    vocab_b = load_vocab(model_b_path)\n    scores["vocab"] = len(vocab_a & vocab_b) / len(vocab_a | vocab_b)\n    \n    # 3. Weight spectrum correlation\n    svd_a = extract_svd_spectra(model_a_path)\n    svd_b = extract_svd_spectra(model_b_path)\n    scores["spectral"] = np.mean([\n        np.corrcoef(svd_a[layer], svd_b[layer])[0,1]\n        for layer in svd_a.keys()\n    ])\n    \n    weights = {"arch": 0.2, "vocab": 0.2, "spectral": 0.2, "activation": 0.2, "loss": 0.2}\n    total = sum(scores[k] * weights[k] for k in weights)\n    \n    return {"total": total, "recommendation": "SAFE" if total > 0.8 else "CAUTION" if total > 0.6 else "RISKY"}`,
    outputData: 'Compatibility score (0-1), per-dimension breakdown, recommendation',
    relevance: 'Run BEFORE merging. Score >0.8 = safe, 0.6-0.8 = caution, <0.6 = risky',
    difficulty: 'expert'
  }
];

export const mergingStrategies = [
  {
    name: 'SLERP (Spherical Linear Interpolation)',
    description: 'Interpolate weights along shortest path on hypersphere.',
    when: 'Compatibility score > 0.8, same architecture family',
    formula: 'w_merged = sin((1-t)Ω)/sin(Ω) * w_a + sin(tΩ)/sin(Ω) * w_b',
    risk: 'Low when models are compatible'
  },
  {
    name: 'TIES (Trim, Elect, Merge)',
    description: 'Trim small weights, elect sign by majority, merge aligned parameters.',
    when: 'Models trained on different data, some capability conflicts',
    formula: 'τ(w) = w if |w| > threshold else 0; sign = mode(signs); merge = mean(w * sign)',
    risk: 'Medium — may lose specialized knowledge'
  },
  {
    name: 'DARE (Drop And REscale)',
    description: 'Randomly drop delta weights, rescale remaining.',
    when: 'Merging many models, preserve diverse capabilities',
    formula: 'δ = w - w_base; δ_dropped = δ * mask(p); w_merged = w_base + δ_dropped / p',
    risk: 'Low-medium — probabilistic'
  },
  {
    name: 'Passthrough (Layer-level)',
    description: 'Take entire layers from different models.',
    when: 'Models have complementary layer strengths',
    formula: 'layer_i = model_a.layer_i if i in strong_layers_a else model_b.layer_i',
    risk: 'High — requires careful layer selection'
  },
  {
    name: 'Frankenmerge (Block-level)',
    description: 'Combine early layers from one model with late layers from another.',
    when: 'One model has better base understanding, other has specialized knowledge',
    formula: 'layers[:cutoff] = model_a; layers[cutoff:] = model_b',
    risk: 'High — cutoff point is critical'
  }
];
