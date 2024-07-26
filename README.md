[![LinkedIn][linkedin-shield]][linkedin-url]

## :jigsaw: Objective

- Implement CLIP Architecture from scratch
- Train model to connect images and captions

## Prerequisites
* [![Python][Python.py]][python-url]
* [![Pytorch][PyTorch.tensor]][torch-url]
* [![Hugging face][HuggingFace.transformers]][huggingface-url]

## :open_file_folder: Files
- [**dataset.py**](dataset.py)
    - This file contains class for dataset and function for transforms
- [**model.py**](model.py)
    - This file contains model architecture
- [**utils.py**](utils.py)
    - This file contains AvgMeter class and method for learning rate.
- [**train.py**](train.py)
    - This is the main file of this project
    - It uses function available in `dataset.py`, `model.py` and `utils.py`
    - It contains functions to train and test model.
- [**inference.py**](inference.py)
    - It contains functions to inference model and retrive images from a folder.

## :building_construction: Model Architecture
The model is implemented based on Learning Transferable Visual Models From Natural Language
Supervision. The transformer architecture is structured using an image encoder and text encoder.
Our specific model consists of ResNet50 as image encoder and DistilBert as text encoder. To bring 
image and text embeddings to same dimensions we use projection layers. We can compare these
projections to push apart the non-relevant images and texts and pull together images and texts.

**Model Names and Dimensions:**

- model_name: 'resnet50'
- image_embedding: 2048
- text_encoder_model: "distilbert-base-uncased"
- text_embedding: 768


## Installation

1. Clone the repo
```
git clone https://github.com/AkashDataScience/clip_pytorch
```
2. Go inside folder
```
 cd clip_pytorch
```
3. Install dependencies
```
pip install -r requirements.txt
```

## Training

```
# Start training with:
python train.py

```

## Inference

```
# Inference with:
python inference.py

# To inference with new query from jupyter notebook
%run inference.py --query="A wolf sitting on a tree trunk"

```

## Usage 
Please refer to [ERA V2 Session 23](https://github.com/AkashDataScience/ERA-V2/tree/master/Week-23)

## Contact

Akash Shah - akashmshah19@gmail.com  
Project Link - [ERA V2](https://github.com/AkashDataScience/ERA-V2/tree/master)

## Acknowledgments
This repo is developed using references listed below:
* [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020)
* [DOMINO: DISCOVERING SYSTEMATIC ERRORS WITH CROSS-MODAL EMBEDDINGS](https://arxiv.org/pdf/2203.14960)
* [GSCLIP : A Framework for Explaining Distribution Shifts in Natural Language](https://arxiv.org/pdf/2206.15007)
* [UIC-NLP at SemEval-2022 Task 5: Exploring Contrastive Learning for Multimodal Detection of Misogynistic Memes](https://aclanthology.org/2022.semeval-1.109.pdf)
* [cdsBERT - Extending Protein Language Models with Codon Awareness](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10516008/)
* [ENIGMA-51: Towards a Fine-Grained Understanding of Human-Object Interactions in Industrial Scenarios](https://arxiv.org/pdf/2309.14809v2)
* [Simple Implementation of OpenAI CLIP model: A Tutorial](https://towardsdatascience.com/simple-implementation-of-openai-clip-model-a-tutorial-ace6ff01d9f2)
* [Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165)


[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/akash-m-shah/
[Python.py]:https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://www.python.org/
[PyTorch.tensor]: https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white
[torch-url]: https://pytorch.org/
[HuggingFace.transformers]: https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange
[huggingface-url]: https://huggingface.co/