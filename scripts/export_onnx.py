"""
Export Trained Model to ONNX for Browser Inference
====================================================
Converts the trained LoRA adapter + classifier to ONNX format
for use with Transformers.js in the Chrome extension.

Run after training:
    python -m ai_nlp.training.emotion_pipeline.run --epochs 15
    python scripts/export_onnx.py

Output: ai_nlp/training/models/onnx/ (ready for extension)
"""

import os
import sys
import json
import shutil
import logging
from typing import Optional

import torch
import numpy as np
from transformers import (
    XLMRobertaConfig,
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Paths
TRAINED_MODEL_DIR = os.path.join("ai_nlp", "training", "checkpoints", "emotion_model", "best_model")
ONNX_OUTPUT_DIR = os.path.join("ai_nlp", "training", "models", "onnx")
EXTENSION_MODEL_DIR = os.path.join("extension", "public", "models")


def export_to_onnx(
    trained_model_path: str = TRAINED_MODEL_DIR,
    output_dir: str = ONNX_OUTPUT_DIR,
    max_seq_length: int = 128,
) -> Optional[str]:
    """
    Export the trained GoEmotions model to ONNX format.
    
    The process:
    1. Load base XLM-RoBERTa + LoRA adapter + classifier head
    2. Merge LoRA weights into base model
    3. Export to ONNX with dynamic input axes
    4. Copy tokenizer files
    5. Copy to extension's public/models directory
    """
    if not os.path.exists(os.path.join(trained_model_path, "adapter_model.safetensors")):
        logger.error(f"No trained model found at {trained_model_path}")
        logger.error("Run training first: python -m ai_nlp.training.emotion_pipeline.run")
        return None
    
    logger.info(f"Exporting model from {trained_model_path} to ONNX...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(EXTENSION_MODEL_DIR, exist_ok=True)
    
    # === Step 1: Load the full trained model ===
    logger.info("Loading trained model (base + LoRA adapter + classifier)...")
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(trained_model_path)
    logger.info(f"Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # Load base model
    base_model = XLMRobertaForSequenceClassification.from_pretrained(
        "xlm-roberta-base",
        num_labels=9,
        torchscript=True,
    )
    
    # Apply LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(base_model, trained_model_path)
    
    # Load classifier head
    classifier_path = os.path.join(trained_model_path, "classifier.pt")
    if os.path.exists(classifier_path):
        classifier_state = torch.load(classifier_path, map_location="cpu")
        model.classifier.load_state_dict(classifier_state)
        logger.info("Classifier head loaded")
    
    model.eval()
    
    # === Step 2: Merge LoRA weights (for simpler ONNX graph) ===
    logger.info("Merging LoRA weights...")
    merged_model = model.merge_and_unload()
    merged_model.eval()
    
    # === Step 3: Export to ONNX ===
    logger.info("Exporting to ONNX...")
    
    # Create dummy input
    dummy_input_ids = torch.randint(0, len(tokenizer), (1, max_seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_seq_length, dtype=torch.long)
    
    # Export
    onnx_path = os.path.join(output_dir, "model.onnx")
    
    with torch.no_grad():
        torch.onnx.export(
            merged_model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
    
    logger.info(f"ONNX model saved: {onnx_path}")
    logger.info(f"File size: {os.path.getsize(onnx_path) / 1e6:.1f} MB")
    
    # === Step 4: Save tokenizer files ===
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Tokenizer saved to {output_dir}")
    
    # === Step 5: Copy config ===
    # Create model config for Transformers.js
    config = {
        "model_type": "xlm-roberta",
        "num_labels": 9,
        "id2label": {
            "0": "admiration", "1": "anger", "2": "anxiety", "3": "fear",
            "4": "joy", "5": "love", "6": "sadness", "7": "surprise", "8": "neutral"
        },
        "label2id": {
            "admiration": 0, "anger": 1, "anxiety": 2, "fear": 3,
            "joy": 4, "love": 5, "sadness": 6, "surprise": 7, "neutral": 8
        },
        "max_position_embeddings": max_seq_length,
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    # === Step 6: Create emotion mapping for extension ===
    emotion_mapping = {
        "input": {
            "admiration": "joy",         # admiration → joy (primary)
            "anger": "anger",
            "anxiety": "anxiety",
            "fear": "fear",
            "joy": "joy",
            "love": "joy",               # love → joy
            "sadness": "sadness",
            "surprise": "surprise",
            "neutral": "neutral",
        },
        "output_labels": [
            "joy", "anger", "sadness", "anxiety", "fear",
            "surprise", "neutral", "toxic", "sarcastic"
        ],
        "version": "1.0.0",
        "description": "GoEmotions-trained XLM-RoBERTa with 9 coarse emotions"
    }
    with open(os.path.join(output_dir, "emotion_mapping.json"), "w") as f:
        json.dump(emotion_mapping, f, indent=2)
    
    # === Step 7: Copy to extension folder ===
    ext_model_dir = os.path.join(EXTENSION_MODEL_DIR, "goemotions")
    os.makedirs(ext_model_dir, exist_ok=True)
    
    for fname in ["model.onnx", "config.json", "tokenizer.json", "tokenizer_config.json", "emotion_mapping.json"]:
        src = os.path.join(output_dir, fname)
        dst = os.path.join(ext_model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"Copied {fname} to extension: {dst}")
    
    logger.info("=" * 60)
    logger.info("EXPORT COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"ONNX model: {onnx_path}")
    logger.info(f"Extension model: {ext_model_dir}")
    logger.info("\nTo use in extension, update emotionClassifier.ts to load from:")
    logger.info("  chrome.runtime.getURL('models/goemotions/model.onnx')")
    
    return onnx_path


if __name__ == "__main__":
    export_to_onnx()