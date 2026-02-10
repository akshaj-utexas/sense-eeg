import torch
import clip
import nltk
import os
from tqdm import tqdm
from nltk.corpus import brown, stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter, defaultdict

# Setup NLTK
for res in ['punkt', 'brown', 'stopwords', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger_eng']:
    nltk.download(res, quiet=True)

def get_imagenet_vocab(dataset_path):
    """Extracts cleaned vocab from your specific EEG/ImageNet dataset."""
    dataset = torch.load(dataset_path, map_location='cpu')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    word_tag_counts = defaultdict(Counter)

    for item in dataset:
        tokens = nltk.word_tokenize(item.get('caption', "").lower())
        for word, tag in nltk.pos_tag(tokens):
            if word.isalpha() and word not in stop_words and len(word) > 2:
                cat = None
                if tag.startswith('NN'): cat = 'n'
                elif tag.startswith('VB'): cat = 'v'
                elif tag.startswith('JJ'): cat = 'a'
                if cat: word_tag_counts[word][cat] += 1

    final_words = set()
    for word, counts in word_tag_counts.items():
        dom_cat = counts.most_common(1)[0][0]
        final_words.add(lemmatizer.lemmatize(word, pos=dom_cat))
    return list(final_words)

def build_corpus(mode="imagenet", dataset_path=None, output_path="data/corpus.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if mode == "brown":
        print("Building Brown Corpus (50k words)...")
        words = brown.words()
        stop_words = set(stopwords.words('english'))
        clean = [w.lower() for w in words if w.isalpha() and w.lower() not in stop_words and len(w) > 2]
        vocab = [w for w, _ in Counter(clean).most_common(50000)]
    else:
        print(f"Building ImageNet Corpus from {dataset_path}...")
        vocab = get_imagenet_vocab(dataset_path)

    # CLIP Encoding (ViT-L/14 to match DreamDiffusion)
    model, _ = clip.load("ViT-L/14", device=device)
    all_embs = []
    
    print(f"Encoding {len(vocab)} words...")
    for i in tqdm(range(0, len(vocab), 512)):
        batch = vocab[i:i+512]
        tokens = clip.tokenize(batch).to(device)
        with torch.no_grad():
            embs = model.encode_text(tokens)
            embs /= embs.norm(dim=-1, keepdim=True)
            all_embs.append(embs.cpu())

    output_path = "data/" + mode + "_corpus.pt"
    torch.save({"words": vocab, "embeddings": torch.cat(all_embs)}, output_path)
    print(f"✅ Corpus saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    build_corpus(mode="imagenet", dataset_path="data/eeg_5_95_text_dataset_train.pth")