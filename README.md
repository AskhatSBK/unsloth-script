# Gemma-3 Uzbek Fine-Tuning Project

This project is designed for fine-tuning the Gemma-3 model using the Uzbek language. It utilizes various libraries to facilitate the training process and manage datasets effectively.

## Project Structure

```
gemma3_uzbek_lora
├── src
│   └── gemma3-27b-finetuning.py
├── checkpoints
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd gemma3_uzbek_lora
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure your Weights & Biases (WandB) API key for tracking experiments.

## Usage

To start the fine-tuning process, run the following command:
```
python src/gemma3-27b-finetuning.py
```

Make sure to adjust any parameters in the script as necessary for your specific use case.

## License

This project is licensed under the MIT License.