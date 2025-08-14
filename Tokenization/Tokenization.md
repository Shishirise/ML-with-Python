# Defination:

In machine learning, especially in natural language processing (NLP),tokenization is the process of breaking text into smaller units called tokens so the computer can work with them more easily.

# What is Tokenization?

A token can be:
A word ("Machine", "learning")
A subword ("learn", "ing")
A character ("M", "a", "c")
Tokenization turns text into a sequence of tokens so that ML models can map them to numbers (embeddings).

Example:
```
Sentence: "Machine learning is fun!"
Tokens: ["Machine", "learning", "is", "fun", "!"]
```

# Tokenization Explained

## 1️⃣ Step 1 — Original Text
```
"Machine learning is fun!"
```

---

## 2️⃣ Step 2 — Tokenization
Depending on the tokenizer type:

**Word-level tokenizer:**
```
["Machine", "learning", "is", "fun", "!"]
```

**Subword tokenizer (BPE, WordPiece, etc.):**
```
["Machine", "learn", "ing", "is", "fun", "!"]
```

**Character-level tokenizer:**
```
["M", "a", "c", "h", "i", "n", "e", " ", "l", "e", "a", "r", "n", "i", "n", "g", " ", "i", "s", " ", "f", "u", "n", "!"]
```

---

## 3️⃣ Step 3 — Map Tokens to IDs (Vocabulary Lookup)
The tokenizer has a **vocabulary** — basically a big dictionary mapping tokens → numbers.

Example (tiny fake vocabulary for demonstration):
```
{
    "Machine": 101,
    "learn": 42,
    "ing": 7,
    "is": 56,
    "fun": 99,
    "!": 12
}
```

If we tokenize `"Machine learning is fun!"` as **subwords**:
```
Tokens: ["Machine", "learn", "ing", "is", "fun", "!"]
IDs:    [101,       42,      7,     56,     99,   12]
```

---

## 4️⃣ Step 4 — Convert IDs to Embeddings
Models don’t directly work with token IDs — instead, each ID is mapped to a **vector** (list of numbers) in an **embedding matrix**.

Example embedding matrix (just small 3D vectors for demo):
```
ID: 101 → [0.2, 0.7, -0.1]
ID: 42  → [0.8, 0.3,  0.5]
ID: 7   → [-0.4, 0.9, 0.1]
ID: 56  → [0.0, 0.5, -0.2]
ID: 99  → [1.0, 0.1,  0.7]
ID: 12  → [0.3, -0.6, 0.4]
```

So our sequence becomes:
```
[[0.2, 0.7, -0.1],   # Machine
 [0.8, 0.3,  0.5],   # learn
 [-0.4, 0.9,  0.1],  # ing
 [0.0, 0.5, -0.2],   # is
 [1.0, 0.1,  0.7],   # fun
 [0.3, -0.6, 0.4]]   # !

Distance is small,model knows they are related.
```

---

✅ **Summary flow:**
**Text → Tokens → Token IDs → Embedding Vectors → Model Input**
