import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import MarianMTModel, MarianTokenizer

MODEL_PATH = 'AdiKr25/En-fr-WMT14-41_Bleu-41_chrF-63'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Loading model on {DEVICE}...')
tokenizer = MarianTokenizer.from_pretrained(MODEL_PATH)
model = MarianMTModel.from_pretrained(MODEL_PATH, output_attentions=True).to(DEVICE)
model.eval()
print('Model ready!')


def translate(text):
    if not text.strip():
        return '', None

    inputs = tokenizer(
        text.strip(),
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=256
    ).to(DEVICE)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            num_beams=8,
            max_length=256,
            early_stopping=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    translated_ids = output.sequences[0]
    translation = tokenizer.decode(translated_ids, skip_special_tokens=True)

    # ── extract cross-attention ───────────────────────────────────────────────
    cross_attns = output.cross_attentions

    rows = []
    if cross_attns:
        for step in cross_attns:
            last_layer = step[-1]        # last decoder layer
            attn = last_layer[0]         # remove batch
            attn = attn.mean(dim=0)      # average heads
            attn = attn.squeeze(0)       # remove tgt dimension
            rows.append(attn)

        attn_matrix = torch.stack(rows).cpu().numpy()
    else:
        attn_matrix = torch.zeros(
            (len(translated_ids), inputs['input_ids'].shape[1])
        ).numpy()
    # ── token labels ─────────────────────────────────────────────────────────
    src_tokens = [t.replace('▁', '').strip() for t in
                  tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
                  if t != '<pad>']
    tgt_tokens = [t.replace('▁', '').strip() for t in
                  tokenizer.convert_ids_to_tokens(translated_ids)
                  if t not in ('<pad>', tokenizer.eos_token)]

    attn_matrix = attn_matrix[:len(tgt_tokens), :len(src_tokens)]

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(max(6, len(src_tokens)*0.7),
                max(4, len(tgt_tokens)*0.6))
    )

    im = ax.imshow(
        attn_matrix,
        cmap='Blues',
        aspect='auto',
        vmin=0,
        vmax=float(attn_matrix.max()) if attn_matrix.size else 1
    )

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=40, ha='right', fontsize=11)

    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens, fontsize=11)

    ax.set_xlabel("English")
    ax.set_ylabel("French")
    ax.set_title("Attention Heatmap")

    plt.colorbar(im, ax=ax, shrink=0.75)
    plt.tight_layout()

    return translation, fig


gr.Interface(
    fn=translate,
    inputs=gr.Textbox(lines=4, label='🇬🇧 English'),
    outputs=[
        gr.Textbox(lines=4, label='🇫🇷 French'),
        gr.Plot(label='Attention Heatmap'),
    ],
    title='English → French Translator',
    examples=[
        ['The sun sets slowly over the horizon.'],
        ['She opened the old book and began to read.'],
        ['I have always loved the smell of fresh bread.'],
    ],
    theme=gr.themes.Soft()
).launch()