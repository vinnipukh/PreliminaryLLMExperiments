# LLM Sentiment Analysis Experiments

This md contains a comprehensive benchmark report of various Large Language Models (LLMs) evaluated on a human-labeled sentiment analysis dataset. The tests were conducted under optimized "TRSA 1" conditions. The load settings, particularly Flash Attention and max GPU offload, are optimized to utilize the 20GB VRAM the testers GPU.

## Dataset used

You can find the details about the dataset used in these experiments in here : https://huggingface.co/datasets/vinnipukh/TachyonTRSA1

## TRSA 1 Conditions

**What are TRSA1 conditions?** 

TRSA1 conditions are a set of optimal conditions found by heavy testing.

### Full Set of Conditions

**System Prompt:**
```
Sen bir duygu analiz uzmanısın. Kullanıcı yorumlarını analiz et. SADECE JSON formatında yanıt ver.
Etiketler: "POSITIVE", "NEGATIVE", "NEUTRAL".
Örnek: {"label": "POSITIVE"}
```

**Temperature:** 0

**Repeat Penalty:** 1

**Min P Sampling:** 0

**Top P Sampling:** 0

**Structured Output:**
```json
{
  "type": "object",
  "properties": {
    "label": {
      "type": "string",
      "enum": ["POSITIVE", "NEGATIVE", "NEUTRAL"]
    }
  },
  "required": ["label"]
}
```

**Load Settings:**
- **GPU Offload:** Max
- **CPU Thread Pool size:** 6
- **Context Length:** 1024 (Hız ve az ısınma için)
- **Flash Attention:** ON (AMD kart için şart)
- **Evaluation Batch Size:** 512
- **Offload KV Cache to GPU Memory:** On
- **Keep Model in Memory:** On
- **Try mmap():** On

---

## UNSLOTH/GPT-OSS-20B

**Conditions:** TRSA1 conditions

**Quantization:** Q4_K_S

**VRAM USE:** 15.2 GB

**Time/Speed:** 04:53 

### Model Performance Report

**Analiz Edilen Toplam Veri:** 1494

**GENEL DOĞRULUK (Accuracy):** %77.11

**Hatalı / Format Dışı Sayısı:** 0

### Detailed Classification Report

| | **precision** | **recall** | **f1-score** | **support** |
|---|---|---|---|---|
| **NEGATIVE** | 0.81 | 0.94 | 0.87 | 541 |
| **NEUTRAL** | 0.52 | 0.75 | 0.61 | 298 |
| **POSITIVE** | 0.97 | 0.64 | 0.77 | 655 |
| **accuracy** | | 0.77 | | 1494 |
| **macro avg** | 0.76 | 0.78 | 0.75 | 1494 |
| **weighted avg** | 0.82 | 0.77 | 0.78 | 1494 |

**Karmaşıklık Matrisi (Confusion Matrix):**

| **Gerçek \ Tahmin** | **NEGATIVE** | **NEUTRAL** | **POSITIVE** |
|---|---|---|---|
| **NEGATIVE** | 509 | 28 | 4 |
| **NEUTRAL** | 64 | 223 | 11 |
| **POSITIVE** | 57 | 178 | 420 |

---

## QWEN3-14B

**Conditions:** TRSA1 conditions

**Quantization:** Q4_K_M

**VRAM USE:** 12.9 GB

**Time/Speed:** 06:04 

### Model Performance Report

**Analiz Edilen Toplam Veri:** 1494

**GENEL DOĞRULUK (Accuracy):** %86.28

**Hatalı / Format Dışı Sayısı:** 0

### Detailed Classification Report

| | **precision** | **recall** | **f1-score** | **support** |
|---|---|---|---|---|
| **NEGATIVE** | 0.85 | 0.96 | 0.90 | 541 |
| **NEUTRAL** | 0.78 | 0.62 | 0.69 | 298 |
| **POSITIVE** | 0.91 | 0.89 | 0.90 | 655 |
| **accuracy** | | 0.86 | | 1494 |
| **macro avg** | 0.84 | 0.82 | 0.83 | 1494 |
| **weighted avg** | 0.86 | 0.86 | 0.86 | 1494 |

**Karmaşıklık Matrisi (Confusion Matrix):**

| **Gerçek \ Tahmin** | **NEGATIVE** | **NEUTRAL** | **POSITIVE** |
|---|---|---|---|
| **NEGATIVE** | 522 | 13 | 6 |
| **NEUTRAL** | 59 | 185 | 54 |
| **POSITIVE** | 33 | 40 | 582 |

---

## GEMMA 3 12B IT

**Conditions:** TRSA1 Conditions

**Quantization:** Q4_K_M

**VRAM USE:** 12.8 GB

**Time/Speed:** 05:46

### Model Performance Report

**Analiz Edilen Toplam Veri:** 1494

**GENEL DOĞRULUK (Accuracy):** %86.01

**Hatalı / Format Dışı Sayısı:** 0

### Detailed Classification Report

| | **precision** | **recall** | **f1-score** | **support** |
|---|---|---|---|---|
| **NEGATIVE** | 0.90 | 0.92 | 0.91 | 541 |
| **NEUTRAL** | 0.65 | 0.81 | 0.72 | 298 |
| **POSITIVE** | 0.96 | 0.83 | 0.89 | 655 |
| **accuracy** | | 0.86 | | 1494 |
| **macro avg** | 0.84 | 0.85 | 0.84 | 1494 |
| **weighted avg** | 0.88 | 0.86 | 0.86 | 1494 |

**Karmaşıklık Matrisi (Confusion Matrix):**

| **Gerçek \ Tahmin** | **NEGATIVE** | **NEUTRAL** | **POSITIVE** |
|---|---|---|---|
| **NEGATIVE** | 500 | 38 | 3 |
| **NEUTRAL** | 40 | 241 | 17 |
| **POSITIVE** | 17 | 94 | 544 |

---

## Kumru 2B

### Metrics

**Accuracy:** 0.54

**F1 Macro:** 0.52

### Classification Report

| | **precision** | **recall** | **f1-score** | **support** |
|---|---|---|---|---|
| **Negative** | 0.66 | 0.75 | 0.70 | 517 |
| **Neutral** | 0.33 | 0.58 | 0.42 | 297 |
| **Positive** | 0.73 | 0.32 | 0.44 | 538 |
| **accuracy** | | | 0.54 | 1352 |
| **macro avg** | 0.57 | 0.55 | 0.52 | 1352 |
| **weighted avg** | 0.61 | 0.54 | 0.54 | 1352 |

**Confusion Matrix - Kumru 2B Instruct:**

| **Actual \ Predicted** | **POS** | **NEU** | **NEG** |
|---|---|---|---|
| **POS** | 170 | 247 | 121 |
| **NEU** | 43 | 173 | 81 |
| **NEG** | 20 | 108 | 389 |

---

## FALCON-H1R-7B

**Conditions:** TRSA1 Conditions

**Quantization:** Q4_K_S

**VRAM USE:** 8.8 GB

**Time/Speed:** 09:08

### Model Performance Report

**Analiz Edilen Toplam Veri:** 1494

**GENEL DOĞRULUK (Accuracy):** %68.21

**Hatalı / Format Dışı Sayısı:** 0

### Detailed Classification Report

| | **precision** | **recall** | **f1-score** | **support** |
|---|---|---|---|---|
| **NEGATIVE** | 0.76 | 0.68 | 0.72 | 541 |
| **NEUTRAL** | 0.46 | 0.26 | 0.33 | 298 |
| **POSITIVE** | 0.68 | 0.87 | 0.76 | 655 |
| **accuracy** | | 0.68 | | 1494 |
| **macro avg** | 0.63 | 0.60 | 0.60 | 1494 |
| **weighted avg** | 0.67 | 0.68 | 0.66 | 1494 |

**Karmaşıklık Matrisi (Confusion Matrix):**

| **Gerçek \ Tahmin** | **NEGATIVE** | **NEUTRAL** | **POSITIVE** |
|---|---|---|---|
| **NEGATIVE** | 370 | 53 | 118 |
| **NEUTRAL** | 68 | 76 | 154 |
| **POSITIVE** | 47 | 35 | 573 |

---

## Aya expense 8b

**Quantization:** Q4_K_M

**VRAM USAGE:** 10 GB

**Time/Speed:** 05:30

### Model Performance Report

**Analiz Edilen Toplam Veri:** 1494

**GENEL DOĞRULUK (Accuracy):** %83.33

**Hatalı / Format Dışı Sayısı:** 0

### Detailed Classification Report

| | **precision** | **recall** | **f1-score** | **support** |
|---|---|---|---|---|
| **NEGATIVE** | 0.86 | 0.93 | 0.90 | 541 |
| **NEUTRAL** | 0.65 | 0.60 | 0.62 | 298 |
| **POSITIVE** | 0.89 | 0.86 | 0.87 | 655 |
| **accuracy** | | 0.83 | | 1494 |
| **macro avg** | 0.80 | 0.80 | 0.80 | 1494 |
| **weighted avg** | 0.83 | 0.83 | 0.83 | 1494 |

**Karmaşıklık Matrisi (Confusion Matrix):**

| **Gerçek \ Tahmin** | **NEGATIVE** | **NEUTRAL** | **POSITIVE** |
|---|---|---|---|
| **NEGATIVE** | 504 | 28 | 9 |
| **NEUTRAL** | 56 | 178 | 64 |
| **POSITIVE** | 23 | 69 | 563 |

---

## NVIDIA-Nemotron-Nano-12B-v2-Q4_K_S

**Quantization:** Q4_K_S

**Time/Speed:** 06:49

**VRAM USAGE:** 11.2 GB

### Model Performance Report

**Analiz Edilen Toplam Veri:** 1494

**GENEL DOĞRULUK (Accuracy):** %85.27

**Hatalı / Format Dışı Sayısı:** 0

### Detailed Classification Report

| | **precision** | **recall** | **f1-score** | **support** |
|---|---|---|---|---|
| **NEGATIVE** | 0.85 | 0.93 | 0.88 | 541 |
| **NEUTRAL** | 0.71 | 0.67 | 0.69 | 298 |
| **POSITIVE** | 0.92 | 0.88 | 0.90 | 655 |
| **accuracy** | | 0.85 | | 1494 |
| **macro avg** | 0.83 | 0.82 | 0.82 | 1494 |
| **weighted avg** | 0.85 | 0.85 | 0.85 | 1494 |

**Karmaşıklık Matrisi (Confusion Matrix):**

| **Gerçek \ Tahmin** | **NEGATIVE** | **NEUTRAL** | **POSITIVE** |
|---|---|---|---|
| **NEGATIVE** | 501 | 32 | 8 |
| **NEUTRAL** | 58 | 199 | 41 |
| **POSITIVE** | 33 | 48 | 574 |

---

## Summary Tables

### Table 1: General Metrics

| **Model** | **Quantization** | **VRAM USE** | **TIME** | **Accuracy** |
|---|---|---|---|---|
| GPT OSS 20B | Q4_K_S | 15.2 GB | 04:53 | %77.11 |
| QWEN3 14B | Q4_K_M | 12.9 GB | 06:04 | %86.28 |
| GEMMA 3 12B IT | Q4_K_M | 12.8 GB | 05:46 | %86.01 |
| NVIDIA-Nemotron-Nano-12B-v2 | Q4_K_S | 11.2 GB | 06:49 | %85.27 |
| Aya expense 8B | Q4_K_M | 10 GB | 05:30 | %83.33 |
| FALCON-H1R-7B | Q4_K_S | 8.8 GB | 09:08 | %68.21 |
| KUMRU 2B | none | ? | ? | %54.14 |

### Table 2: F1 Scores and Recall

| **Model** | **F1 MACRO** | **F1 WEIGHTED** | **NEG RECALL** | **NEU RECALL** |
|---|---|---|---|---|
| GPT OSS 20B | 0.75 | 0.78 | 0.94 | 0.75 |
| QWEN3 14B | 0.83 | 0.86 | 0.96 | 0.62 |
| GEMMA 3 12B IT | 0.84 | 0.86 | 0.92 | 0.81 |
| NVIDIA-Nemotron-Nano-12B-v2 | 0.82 | 0.85 | 0.93 | 0.67 |
| Aya expense 8B | 0.80 | 0.83 | 0.93 | 0.60 |
| FALCON-H1R-7B | 0.60 | 0.66 | 0.68 | 0.26 |
| KUMRU 2B | 0.52 | 0.54 | 0.75 | 0.58 |



## References

I want to present my thanks to all the people involved in the training and sharing of the following models.I hope there will be more open models in the future.

https://huggingface.co/bartowski/nvidia_NVIDIA-Nemotron-Nano-12B-v2-GGUF

https://huggingface.co/CohereLabs/aya-expanse-8b

https://huggingface.co/tiiuae/Falcon-H1R-7B

https://huggingface.co/vngrs-ai/Kumru-2B

https://huggingface.co/lmstudio-community/gemma-3-12b-it-GGUF

https://huggingface.co/Qwen/Qwen3-14B

https://huggingface.co/unsloth/gpt-oss-20b
