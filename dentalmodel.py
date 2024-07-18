import streamlit as st
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import io
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

st.set_page_config(layout="wide")

@st.cache_resource
def load_model():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def process_image(image_bytes, model):
    img = Image.open(io.BytesIO(image_bytes))
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0)
    with torch.no_grad():
        results = model(img_tensor)
    return results[0]

def plot_results(image, result):
    fig = go.Figure()

    img = Image.open(io.BytesIO(image))
    img_array = np.array(img)

    fig.add_trace(go.Image(z=img_array))

    for box in result['boxes']:
        x0, y0, x1, y1 = box.int().tolist()
        fig.add_shape(
            type="rect",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color="red", width=2),
        )

    fig.update_layout(
        height=600,
        width=800,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
    )

    return fig

def count_teeth(result):
    return len(result['boxes'])

def count_caries(result):
    caries_count = 0
    for label in result['labels']:
        if label == 'caries':  # Assuming 'caries' is the label for cavities
            caries_count += 1
    return caries_count

def count_missing_teeth(total_teeth, detected_teeth):
    return total_teeth - detected_teeth

st.title("Teeth Segmentation and Counting in Panoramic X-ray Images")

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()

    try:
        result = process_image(image_bytes, model)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image_bytes)

        with col2:
            st.subheader("Segmentation Result")
            fig = plot_results(image_bytes, result)
            st.plotly_chart(fig, use_container_width=True)

        # Count the number of segmented teeth, caries, and missing teeth
        num_teeth = count_teeth(result)
        num_caries = count_caries(result)
        total_teeth = 32  # Assuming a full set of teeth
        num_missing_teeth = count_missing_teeth(total_teeth, num_teeth)

        st.subheader("Detection Results")
        st.write(f"Number of teeth detected: {num_missing_teeth}")
        st.write(f"Number of caries detected: {num_caries}")
        st.write(f"Number of missing teeth detected: {num_teeth}")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
        st.error("Please try uploading a different image.")

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("This demo showcases teeth segmentation and counting using a Mask R-CNN model.")