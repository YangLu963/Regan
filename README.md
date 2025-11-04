# RAGEN WebShop Training on Modal

A sophisticated reinforcement learning agent for e-commerce navigation, trained to interact with WebShop environments. This project implements both real WebShop integration and a detailed simulated environment for robust training.

## 🚀 Features

- **Dual Environment Support**: Real WebShop integration with fallback to high-fidelity simulation
- **Advanced Training Metrics**: Comprehensive evaluation with multiple reward components
- **Modal Cloud Deployment**: Scalable training on GPU instances
- **Detailed Analytics**: Step-by-step training visualization and performance tracking
- **Persistent Storage**: Model checkpointing and results persistence

## 📊 Training Architecture

### Environment Components
- **Real WebShop**: Integration with Princeton NLP's WebShop environment
- **Simulated Environment**: High-fidelity e-commerce simulation with:
  - 10+ diverse products across electronics, clothing, and home categories
  - Multiple product attributes (brand, color, storage, size, etc.)
  - Realistic user queries and target matching

### Reward System
- **Base Reward**: Successful product selection
- **Efficiency Bonus**: Fewer steps yield higher rewards
- **Accuracy Bonus**: Correct target product selection
- **Diversity Bonus**: Using varied filter strategies

## 🛠️ Installation & Setup

### Prerequisites
- Modal account and CLI setup
- Python 3.10+
- Git

### Modal Configuration
```bash
# Install Modal CLI
pip install modal

# Configure Modal
modal token new

# Set up secrets (if using HuggingFace)
modal secret create my-huggingface-secret
