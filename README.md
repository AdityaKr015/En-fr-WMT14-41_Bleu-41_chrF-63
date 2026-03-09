# 🇬🇧 → 🇫🇷 English to French Neural Machine Translation

A high-quality Neural Machine Translation (NMT) system trained on the **WMT14 English–French dataset** using a **Transformer-based MarianMT architecture**.

The project fine-tunes the pretrained `Helsinki-NLP/opus-mt-tc-big-en-fr` model and deploys an interactive translation demo using **Gradio + Hugging Face Spaces**.

---

## 🚀 Demo

Live Demo (Gradio App):  
👉 [https://huggingface.co/spaces/<your-space>](https://huggingface.co/spaces/AdiKr25/En-fr-WMT14-41_Bleu-41_chrF-63)

Model on HuggingFace:  
👉 [https://huggingface.co/AdiKr25/En-fr-WMT14-41_Bleu-41_chrF-63](https://huggingface.co/AdiKr25/En-fr-WMT14-41_Bleu-41_chrF-63)

---

# 🧠 Model Architecture

Transformer based sequence-to-sequence model:

<img width="462" height="733" alt="model_architecture" src="https://github.com/user-attachments/assets/82e3ba84-d82a-4375-9b91-b99716689668" />


Base model:

Helsinki-NLP/opus-mt-tc-big-en-fr


---

# 📊 Dataset

Dataset: **WMT14 English-French**

Initial dataset size:

40,836,715 sentence pairs


Subset used for training:

80,000 pairs


After cleaning:

74,359 pairs


Train / Validation / Test split:

| Split | Samples |
|------|--------|
| Train | 66,923 |
| Validation | 3,718 |
| Test | 3,718 |

---

# 🧹 Data Cleaning

Filtering rules applied:

- Remove sentences shorter than **3 words**
- Remove sentences longer than **128 tokens**
- Remove extreme length mismatches
- English/French ratio constraint


0.5 < length_ratio < 1.5


---

# ⚙️ Training Configuration

| Parameter | Value |
|--------|------|
| Base Model | opus-mt-tc-big-en-fr |
| Epochs | 7 |
| Learning Rate | 2e-5 |
| Batch Size | 32 |
| Gradient Accumulation | 2 |
| Effective Batch | 64 |
| Scheduler | Cosine |
| FP16 | Enabled |
| Beam Size | 8 |

Training hardware:


Tesla P100 GPU


Training time:


~4.5 hours


---

# 📈 Results

### Validation Metrics

| Metric | Score |
|------|------|
| BLEU | **41.76** |
| BLEU-1 | 65.29 |
| BLEU-2 | 46.58 |
| BLEU-3 | 36.39 |
| BLEU-4 | 29.19 |
| Loss | 2.38 |

### Test Metrics

| Metric | Score |
|------|------|
| BLEU | **41.13** |
| chrF | **63.56** |
| TER | 52.44 |

---

# 📊 Training Curves

Loss and BLEU progression during training:

![Training Curves](images/training_curves.png)

---

# 📊 Dataset Statistics

Sentence length distribution:

![Dataset Distribution](images/eda.png)

---

# 🖥️ Interactive Demo

The project includes a **Gradio interface** deployed on HuggingFace Spaces.

Features:

✔ English → French translation  
✔ Beam search decoding  
✔ Attention visualization heatmap  

Example:


Input:
The sun sets slowly over the horizon.

Output:
Le soleil se couche lentement à l'horizon.


The attention visualization helps interpret which source words influence each translated token.

---

# 📦 Installation

```bash
pip install -r requirements.txt
```

Dependencies include:

transformers
datasets
torch
evaluate
sacrebleu
gradio
sentencepiece

▶️ Run the App
```bash
python app.py
```

The app loads the model from HuggingFace Hub and launches a Gradio interface.

The interface implementation is in:

app.py

Example from the translation pipeline:

model.generate(
    num_beams=8,
    max_length=256
)

The model is loaded from the HuggingFace model hub:

https://huggingface.co/AdiKr25/En-fr-WMT14-41_Bleu-41_chrF-63

📂 Project Structure
NMT-English-French/
│
├── app.py
├── requirements.txt
├── README.md
│
├── notebooks/
│   └── training.ipynb
│
├── images/
│   ├── eda.png
│   └── training_curves.png
│
└── results/
    └── training_summary.json
    
🔬 Evaluation Metrics

Evaluation performed using:

SacreBLEU
chrF
TER

Metrics computed using the evaluate and sacrebleu libraries.

Training and evaluation pipeline code is available in the notebook:


🔮 Future Improvements

Possible improvements:

• Train on 200k+ samples
• Increase max sequence length to 256
• Use LoRA fine-tuning for efficiency
• Train a custom transformer instead of fine-tuning

Expected improvement:

+3 to +5 BLEU
📜 License

MIT License

👨‍💻 Author

Aditya
AI / ML Student

Project created for learning Neural Machine Translation and Transformer training pipelines.


That’s the part most people completely mess up.
Sources
