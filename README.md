# Tuning Lab

A collection of fine-tuning techniques for Large Language Models (LLMs) implemented from scratch. This repository serves as both an educational resource and a practical toolkit for understanding and applying various LLM fine-tuning methods.

## Overview

This project aims to provide clear, well-documented implementations of popular fine-tuning techniques, built from the ground up without relying on high-level abstractions. Each implementation focuses on clarity and educational value while maintaining practical usability.

## Current Implementation

### Reinforcement Learning - PPO

**PPO (Proximal Policy Optimization)** - A reinforcement learning algorithm used for RLHF (Reinforcement Learning from Human Feedback) to align language models with human preferences.

PPO is widely used in training models like ChatGPT and Claude to follow instructions and generate helpful, harmless, and honest responses.

## Project Structure

```
tuning-lab/
├── rl/
│   └── ppo/
│       ├── model.py
│       └── README.md
└── README.md
```

## Getting Started

Navigate to the `rl/ppo/` directory for implementation details, usage examples, and requirements.

## Requirements

- Python 3.12
- PyTorch
- Additional dependencies listed in each technique's directory

## Contributing

Contributions are welcome! Whether it's bug fixes, optimizations, or documentation improvements, feel free to open an issue or submit a pull request.

## License

MIT License

## References

Relevant papers and resources are listed in the PPO directory's README.