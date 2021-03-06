import numpy as np
import streamlit as st
from PIL import Image
import torch
import clip
from torchray.attribution.grad_cam import grad_cam
# from utils import imgviz
from miniclip.imageWrangle import heatmap, min_max_norm, torch_to_rgba

st.set_page_config(layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache(show_spinner=True, allow_output_mutation=True)
def get_model():
    model, preprocess = clip.load("RN50", device=device, jit=False)
    return model, preprocess

# OPTIONS:
st.sidebar.header('Options')
alpha = st.sidebar.radio("select alpha", [0.5, 0.7, 0.8], index=1)
layer = st.sidebar.selectbox("select saliency layer", ['layer4.2.relu'], index=0)

st.header("Saliency Map demo for CLIP")
st.write("a quick demo by [Hendrik Strobelt](http://hendrik.strobelt.com) (MIT-IBM AI Lab) ")
with st.beta_expander('1. Upload Image', expanded=True):
    imageFile = st.file_uploader("Select a file:", type=[".jpg", ".png", ".jpeg"])

# st.write("### (2) Enter some desriptive texts.")
with st.beta_expander('2. Write Descriptions', expanded=True):
    textarea = st.text_area("Descriptions seperated by semicolon","a car; a dog; a cat")
    prefix = st.text_input("(optional) Prefix all descriptions with: ", "an image of")



if imageFile:
    st.markdown("<hr style='border:black solid;'>", unsafe_allow_html=True)
    image_raw = Image.open(imageFile)
    model, preprocess = get_model()

    # preprocess image:
    image = preprocess(image_raw).unsqueeze(0).to(device)

    #preprocess text
    prefix = prefix.strip()
    if len(prefix)>0:
        categories = [f"{prefix} {x.strip()}" for x in textarea.split(';')]
    else:    
        categories = [x.strip() for x in textarea.split(';')]
    text = clip.tokenize(categories).to(device)
    # st.write(text)
    # with st.echo():
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features_norm = image_features.norm(dim=-1, keepdim=True)
        image_features_new = image_features / image_features_norm
        text_features_norm = text_features.norm(dim=-1, keepdim=True)
        text_features_new = text_features / text_features_norm
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features_new @ text_features_new.t()
        probs = logits_per_image.softmax(dim=-1).cpu().numpy().tolist()

    saliency = grad_cam(model.visual, image.type(model.dtype), image_features, saliency_layer=layer)
    hm = heatmap(image[0], saliency[0][0,].detach().type(torch.float32).cpu(), alpha=alpha)

    collect_images = []
    for i in range(len(categories)):
        # mutliply the normalized text embedding with image norm to get approx image embedding
        text_prediction = (text_features_new[[i]] * image_features_norm)
        saliency = grad_cam(model.visual, image.type(model.dtype), text_prediction, saliency_layer=layer)
        hm = heatmap(image[0], saliency[0][0,].detach().type(torch.float32).cpu(),alpha=alpha)
        collect_images.append(hm)
    logits = logits_per_image.cpu().numpy().tolist()[0]
    st.write("### Grad Cam for text embeddings")
    st.image(collect_images, 
            width=256,
            caption=[f"{x} - {str(round(y,3))}/{str(round(l,2))}" for (x,y,l) in zip(categories, probs[0], logits)])

    st.write("### Original Image and Grad Cam for image embedding")
    st.image([Image.fromarray((torch_to_rgba(image[0]).numpy()*255.).astype(np.uint8)),hm], caption=["original","image gradcam"]) #, caption="Grad Cam for original embedding")


    # st.image(imageFile)

# @st.cache
def get_readme():
    with open('README.md') as f:
        return "\n".join([x.strip() for x in f.readlines()])
st.markdown("<hr style='border:black solid;'>", unsafe_allow_html=True) 
st.markdown(get_readme(), unsafe_allow_html=True)
