import os
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models

from PIL import Image
from torchvision import transforms


# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(
    page_title="Fashion Recommender",
    page_icon="👗",
    layout="wide"
)

st.title("👗 Fashion Item Recommender")
st.write("Find similar fashion items using ResNet18 feature extraction.")


# =========================================================
# CONSTANTS
# =========================================================
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

TRAIN_CSV   = "data/fashion-mnist_train.csv"
TEST_CSV    = "data/fashion-mnist_test.csv"
MODEL_PATH  = "saved_models/resnet18_fashionmnist.pth"

# Disk cache — lives next to this script so it survives Streamlit re-runs
CACHE_DIR   = "feature_cache"
FEAT_FILE   = os.path.join(CACHE_DIR, "train_features.npy")
LABEL_FILE  = os.path.join(CACHE_DIR, "train_labels.npy")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 256   # larger batch → fewer forward passes → much faster


# =========================================================
# IMAGE TRANSFORM
# =========================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# =========================================================
# LOAD MODEL  (cached for the whole session)
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


# =========================================================
# FEATURE EXTRACTOR  (strip final FC + flatten)
# =========================================================
@st.cache_resource(show_spinner=False)
def get_extractor():
    model = load_model()
    extractor = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    extractor.to(DEVICE)
    extractor.eval()
    return extractor


# =========================================================
# LOAD CSV DATA  (cached for the whole session)
# =========================================================
@st.cache_data(show_spinner=False)
def load_data():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)
    return train_df, test_df


# =========================================================
# EXTRACT & CACHE TRAIN FEATURES
#   • First run  → compute + save to disk  (~1-2 min)
#   • Every subsequent run → load from disk (~1-2 sec)
# =========================================================
def get_train_features(train_df):
    # ---- fast path: load from disk ----------------------
    if os.path.exists(FEAT_FILE) and os.path.exists(LABEL_FILE):
        features = np.load(FEAT_FILE)
        labels   = np.load(LABEL_FILE)
        return features, labels

    # ---- slow path: compute once and persist ------------
    os.makedirs(CACHE_DIR, exist_ok=True)
    extractor = get_extractor()

    pixel_cols = train_df.iloc[:, 1:].values.astype(np.uint8)   # (N, 784)
    label_col  = train_df.iloc[:, 0].values                     # (N,)
    total      = len(train_df)

    st.info("⏳ First-time setup: extracting features for all 60 000 training "
            "images. This takes ~1–2 minutes and is saved to disk — "
            "**you will never wait again after this.**")

    bar = st.progress(0, text="Extracting features…")
    all_feats = []

    for start in range(0, total, BATCH):
        end       = min(start + BATCH, total)
        batch_pix = pixel_cols[start:end]             # (B, 784)

        # Build batch tensor without per-image Python loop
        imgs = batch_pix.reshape(-1, 28, 28)          # (B, 28, 28)
        tensors = []
        for pix in imgs:
            img    = Image.fromarray(pix, mode="L").convert("RGB")
            tensors.append(transform(img))

        batch_tensor = torch.stack(tensors).to(DEVICE)

        with torch.no_grad():
            feats = extractor(batch_tensor).cpu().numpy()

        all_feats.append(feats)
        bar.progress(end / total, text=f"Extracting features… {end}/{total}")

    bar.empty()

    features = np.vstack(all_feats).astype(np.float32)
    labels   = label_col.astype(np.int32)

    # Pre-normalise once so similarity queries are just a dot-product
    norms    = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.where(norms == 0, 1, norms)

    np.save(FEAT_FILE, features)
    np.save(LABEL_FILE, labels)
    st.success("✅ Features saved to disk. Future loads will be instant!")

    return features, labels


# =========================================================
# COSINE SIMILARITY  (features already unit-normalised)
# =========================================================
def cosine_similarity_topk(query_feat, train_feats, k):
    q_norm  = query_feat / (np.linalg.norm(query_feat) or 1.0)
    scores  = train_feats @ q_norm                 # dot-product on unit vecs
    top_idx = np.argpartition(scores, -k)[-k:]     # O(N) partial sort
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
    return top_idx, scores[top_idx]


# =========================================================
# LOAD DATA
# =========================================================
train_df, test_df = load_data()


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Settings")

top_k = st.sidebar.slider("Top-K Recommendations", 3, 10, 5)

selected_class = st.sidebar.selectbox("Choose Class", CLASS_NAMES)
cls_id         = CLASS_NAMES.index(selected_class)

class_rows = test_df[test_df.iloc[:, 0] == cls_id].reset_index(drop=True)

sample_idx = st.sidebar.slider(
    "Sample Number", 0, min(49, len(class_rows) - 1), 0
)

# Cache-invalidation button
if st.sidebar.button("🗑️ Clear Feature Cache", help="Force re-extraction"):
    for f in (FEAT_FILE, LABEL_FILE):
        if os.path.exists(f):
            os.remove(f)
    st.cache_data.clear()
    st.sidebar.success("Cache cleared. Reload to re-extract.")

row         = class_rows.iloc[sample_idx]
query_pixels = row.iloc[1:].values.astype(np.uint8).reshape(28, 28)
query_img   = Image.fromarray(query_pixels, mode="L")


# =========================================================
# QUERY IMAGE
# =========================================================
st.subheader("🔎 Query Image")
col1, col2 = st.columns([1, 3])

with col1:
    st.image(query_img.resize((140, 140)), caption=selected_class)

with col2:
    st.write(f"**Class:** {selected_class}")
    st.write(f"**Class ID:** {cls_id}")
    st.write("Press the button below to find similar items.")


# =========================================================
# FIND SIMILAR ITEMS
# =========================================================
if st.button("🔍 Find Similar Items"):

    # Load / compute features (instant after first run)
    train_feats, train_labels = get_train_features(train_df)

    # Extract query feature
    extractor   = get_extractor()
    query_tensor = transform(query_img.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        query_feat = extractor(query_tensor).squeeze().cpu().numpy()

    top_idx, top_scores = cosine_similarity_topk(query_feat, train_feats, top_k)
    top_labels          = train_labels[top_idx]

    # Precision
    correct   = int(np.sum(top_labels == cls_id))
    precision = correct / top_k

    # =====================================================
    # RESULTS
    # =====================================================
    st.divider()
    st.subheader(f"✅ Top-{top_k} Similar Items")

    prec_col, _ = st.columns([2, 3])
    with prec_col:
        st.metric(
            label=f"Precision@{top_k}",
            value=f"{precision * 100:.0f}%",
            delta=f"{correct}/{top_k} correct"
        )
    st.progress(precision)

    cols = st.columns(top_k)
    for i, (idx, score, lbl) in enumerate(zip(top_idx, top_scores, top_labels)):
        pixels  = train_df.iloc[idx, 1:].values.astype(np.uint8).reshape(28, 28)
        rec_img = Image.fromarray(pixels, mode="L")

        with cols[i]:
            st.image(rec_img.resize((100, 100)))
            st.write(f"**#{i+1}** {CLASS_NAMES[lbl]}")
            st.caption(f"Similarity: {score:.3f}")
            if lbl == cls_id:
                st.success("✓ Match")
            else:
                st.error("✗ Different")