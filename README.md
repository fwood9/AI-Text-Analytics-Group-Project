# AI-Text-Analytics-Group-Project
# Dice Image Captioning Project

## Project Overview


## Dataset
The dataset is generated automatically using a custom script. It includes variations in:
- Number of dice (1–3)
- Dice values (1–6)
- Colour, size, and rotation

The dataset is split into training, validation, and test sets.

## Project Structure
- Data_generation.ipynb: generate images and corresponding text descriptions
- finished_notebook.ipynb: main experiment pipeline
- eval.py: shared evaluation functions
- outputs/: experiment results, logs and figures
- Working/: development notebooks and intermediate work

## Methods
The project explores two main axes:
- Text representation (e.g. one-hot, TF-IDF, SBERT)
- CNN architectures (e.g. custom CNN, ResNet18, EfficientNet)

A comparison with LLM-based methods is also included.

## Evaluation
We evaluate the model using:
- Exact match accuracy
- Cosine similarity
- BLEU score

## How to Run
1. Run `Data_generation.ipynb` to generate the dataset
2. Run training notebooks or scripts in the repository
3. Use `eval.py` to evaluate results

## Team Contributions
- Hongze: Dataset generation (image and text pipeline)
- Sakshi: Evaluation framework, LLM comparison, report lead
- Liang: Text representation experiments
- Swetha: CNN architectures and training
- Finn: Error analysis and report writing

## Notes
Code is managed using a centralised workflow, where one team member is responsible for organising and uploading the final code to GitHub.
