import gc
import cv2
import torch
import argparse
import torch.nn.functional as F
from tqdm import tqdm
from transformers import DistilBertTokenizer
import matplotlib.pyplot as plt

import config as CFG
from train import build_loaders, make_train_valid_dfs
from model import CLIPModel

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CLIP Training')
    parser.add_argument('--weights_path', default='best.pt', type=str, help="Path of trained model weights")
    parser.add_argument('--query', default='a group of people dancing in a party', type=str,
                        help="Query to retrive image")
    args = parser.parse_args()
    return args

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

def find_matches(model, image_embeddings, query, image_filenames, n=9):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    _, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = cv2.imread(f"{CFG.image_path}/{match}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)
        ax.axis("off")
    
    plt.show()

def infer():
    args = get_args()
    _, valid_df = make_train_valid_dfs()
    model, image_embeddings = get_image_embeddings(valid_df, args.weights_path)
    find_matches(model, image_embeddings, query=args.query, image_filenames=valid_df['image'].values,
                 n=9)
    
if __name__ == '__main__':
    infer()