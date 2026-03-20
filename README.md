# LogicSynth Transformer
A decoder-only Transformer architecture optimized for length-generalization in arithmetic reasoning and logic tasks.

## 📌 Project Overview
LogicSynth explores the boundary of length-generalization in Transformers. By combining Rotary Positional Embeddings (RoPE) with a Curriculum Learning strategy, the model is designed to generalize from shorter training sequences (e.g., 10-digit) to longer inference sequences (e.g., 15-digit) without the need for retraining.

This work serves as a proof of concept for processing long-sequence signal data with complex carry-over logic, drawing parallels to temporal correlations found in high-energy physics data, such as ATLAS calorimeter pulse shapes.

## 🚀 Key Technical Features
Architecture: Decoder-only Transformer.

Positional Encoding: Rotary Positional Embeddings (RoPE) for better relative position handling and extrapolation.

Training Strategy: Curriculum Learning (gradually increasing sequence complexity) to improve convergence and generalization.

Generalization Goal: Arithmetic reasoning with significant length expansion during inference.

## 🚧 Status: Work in Progress
This repository is a prototype.

Current State: Core architecture with RoPE and Curriculum training pipeline is functional.

Active Development: Fine-tuning the length-generalization limits and optimizing carry-over logic accuracy.

## 📁 File Structure
model.py: Implementation of the decoder-only Transformer with RoPE.

train.py: Training logic including the Curriculum Learning scheduler.

dataset.py: Generation and preprocessing of arithmetic/logic sequences.

eval.py: Evaluation scripts specifically testing length-generalization.

main.py: Entry point for execution.

## 🛠️ Future Roadmap (GSoC Goals)
Scale the curriculum to handle even higher-order digit reasoning.

Perform comparative analysis between RoPE and standard ALiBi/Absolute embeddings.

Implement a more robust verification suite for complex signal-data analogies.
