from segment_anything import SamPredictor, sam_model_registry
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F
def get_predictor(model_type='vit_b',model_path=''):
    sam = sam_model_registry[model_type](checkpoint=model_path)
    predictor = SamPredictor(sam)
    return predictor
def get_embedding(predictor,image_path):
    img=Image.open(image_path).convert("RGB")
    img=np.array(img)
    predictor.set_image(img)
    return predictor.features


def get_simi(embedding, h, w, clear_range=5):
    """
    Calculate cosine similarity between a source embedding and all other embeddings in a spatial grid,
    excluding a specified range around the source.

    Args:
    embedding (torch.Tensor): A 4D tensor of embeddings (1, embedding_dim, height, width).
    h (int): Height index of the source embedding.
    w (int): Width index of the source embedding.
    clear_range (int): Range around the source to exclude from calculation.

    Returns:
    np.array: A heatmap of cosine similarities.
    """

    # Extract the source embedding
    src = embedding[0, :, h, w].unsqueeze(0)

    # Initialize an array to hold cosine similarities
    cos_simi = np.zeros((64, 64))

    # Calculate cosine similarity for each position
    min_val=1
    max_val=0
    for i in range(64):
        for j in range(64):
            if h - clear_range <= i <= h + clear_range and w - clear_range <= j <= w + clear_range:
                continue
            target = embedding[0, :, i, j].unsqueeze(0)
            cos_simi[i, j] = F.cosine_similarity(src, target, dim=1).item()
            min_val = min(cos_simi[i,j],min_val)
            max_val = max(cos_simi[i,j],max_val)
    # Min-max normalization
    cos_simi = (cos_simi - min_val) / (max_val-min_val)
    cos_simi[cos_simi<0]=0
    return cos_simi
    # min-max norm
    # return the heatmap
def visual(heatmap, image_path, save_path):
    # Load and resize the image
    image = Image.open(image_path).convert("RGBA")
    image_size = image.size

    # Resize heatmap to image size using nearest interpolation
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(image_size, resample=Image.NEAREST))

    # Convert heatmap to RGBA format
    heatmap_rgba = np.zeros((heatmap_resized.shape[0], heatmap_resized.shape[1], 4), dtype=np.uint8)
    heatmap_rgba[..., 0] = 255  # Red channel
    heatmap_rgba[..., 3] = (heatmap_resized * 255).astype(np.uint8)  # Alpha channel

    # Convert heatmap to an image
    heatmap_image = Image.fromarray(heatmap_rgba)

    # Overlay the heatmap onto the original image
    composite = Image.alpha_composite(image, heatmap_image)

    # Convert back to RGB and save
    composite.convert("RGB").save(save_path)
img_list=os.listdir('./dataset')
predictot=get_predictor(model_type='vit_b',model_path='./model_path/sam_vit_b_01ec64.pth')
for image_name in img_list:
    image_path= os.path.join('./dataset',image_name)
    embed= get_embedding(predictot,image_path)
    heatmap=get_simi(embed,20,20)
    heatmap[20,20]=1
    visual(heatmap,image_path,'./res.jpg')