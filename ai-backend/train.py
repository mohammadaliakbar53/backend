"""
Training Script for SindhAgriNet in PyTorch
Trains the model on multilingual Sindh agricultural datasets and outputs checkpoint and vocabulary.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    TRAINING_SAMPLES,
    DISEASE_CLASSES,
    INTENT_CLASSES,
    CROP_CLASSES,
    SindhTokenizer,
    SindhAgronomyDataset
)
from model import SindhAgriNet

def train_model():
    print("=" * 60)
    print("Starting Sindh Smart Farmer AI - PyTorch Model Training")
    print("=" * 60)

    # 1. Prepare Tokenizer & Vocabulary
    all_texts = [sample["text"] for sample in TRAINING_SAMPLES]
    tokenizer = SindhTokenizer()
    vocab = tokenizer.build_vocab(all_texts, min_freq=1)
    vocab_size = len(vocab)
    print(f"[*] Vocabulary size: {vocab_size} tokens (Urdu, Sindhi, English)")

    # Save vocabulary
    script_dir = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(script_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"[*] Saved vocabulary to {vocab_path}")

    # 2. Prepare Dataset and DataLoader
    # Augment dataset by repeating samples to simulate batch learning
    augmented_samples = TRAINING_SAMPLES * 8
    dataset = SindhAgronomyDataset(augmented_samples, tokenizer, max_len=32)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    print(f"[*] Total training instances (augmented): {len(dataset)}")

    # 3. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    model = SindhAgriNet(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_diseases=len(DISEASE_CLASSES),
        num_intents=len(INTENT_CLASSES),
        num_crops=len(CROP_CLASSES),
        num_langs=3
    ).to(device)

    # Loss Functions & Optimizer
    criterion_disease = nn.CrossEntropyLoss()
    criterion_intent = nn.CrossEntropyLoss()
    criterion_crop = nn.CrossEntropyLoss()
    criterion_lang = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    num_epochs = 25
    history = []

    model.train()
    for epoch in range(1, num_epochs + 1):
        total_loss = 0.0
        correct_disease = 0
        correct_intent = 0
        total_items = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            disease_label = batch["disease_label"].to(device)
            intent_label = batch["intent_label"].to(device)
            crop_label = batch["crop_label"].to(device)
            lang_label = batch["lang_label"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids)

            l_dis = criterion_disease(outputs["disease_logits"], disease_label)
            l_int = criterion_intent(outputs["intent_logits"], intent_label)
            l_crop = criterion_crop(outputs["crop_logits"], crop_label)
            l_lang = criterion_lang(outputs["lang_logits"], lang_label)

            loss = l_dis + 0.6 * l_int + 0.5 * l_crop + 0.4 * l_lang
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(input_ids)
            preds_dis = outputs["disease_logits"].argmax(dim=1)
            preds_int = outputs["intent_logits"].argmax(dim=1)
            correct_disease += (preds_dis == disease_label).sum().item()
            correct_intent += (preds_int == intent_label).sum().item()
            total_items += len(input_ids)

        scheduler.step()

        epoch_loss = total_loss / total_items
        acc_disease = (correct_disease / total_items) * 100.0
        acc_intent = (correct_intent / total_items) * 100.0

        history.append({
            "epoch": epoch,
            "loss": round(epoch_loss, 4),
            "disease_accuracy": round(acc_disease, 2),
            "intent_accuracy": round(acc_intent, 2)
        })

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{num_epochs}] | Loss: {epoch_loss:.4f} | Disease Acc: {acc_disease:.1f}% | Intent Acc: {acc_intent:.1f}%")

    # 4. Save Model Checkpoint & Metadata
    model_path = os.path.join(script_dir, "sindh_cropnet_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"[+] Model checkpoint successfully saved: {model_path}")

    # Total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    meta = {
        "model_name": "SindhAgriNet-PyTorch-v1.0",
        "parameters": total_params,
        "epochs": num_epochs,
        "final_loss": history[-1]["loss"],
        "final_accuracy": history[-1]["disease_accuracy"],
        "disease_classes": DISEASE_CLASSES,
        "intent_classes": INTENT_CLASSES,
        "crop_classes": CROP_CLASSES,
        "languages": ["English", "Urdu", "Sindhi"],
        "history": history
    }

    meta_path = os.path.join(script_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[+] Model metadata successfully saved: {meta_path}")
    print("=" * 60)
    print(f"PyTorch Training Completed Successfully! Total Params: {total_params:,}")
    print("=" * 60)

if __name__ == "__main__":
    train_model()
