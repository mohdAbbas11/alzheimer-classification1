import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import requests
from pathlib import Path

# Import models
from test import EnhancedAlzheimerNet, categories, device, data_transforms
from gan_models import CGAN, CycleGAN, HighResGAN

# Set page config
st.set_page_config(
    page_title="Alzheimer's Disease Detection & Image Generation",
    page_icon="🧠",
    layout="wide"
)

# Create directories if they don't exist
os.makedirs("generated_images", exist_ok=True)
os.makedirs("uploaded_images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Define model paths - use relative paths for GitHub deployment
MODEL_PATHS = {
    'resnet101': os.path.join("models", "best_resnet101.pth"),
    'resnext101_32x8d': os.path.join("models", "best_resnext101_32x8d.pth"),
    'densenet161': os.path.join("models", "best_densenet161.pth"),
    'ensemble': os.path.join("models", "best_alzheimer_model.pth"),
    # GAN model paths
    'cgan_generator': os.path.join("models", "cgan_generator.pth"),
    'cyclegan_G_AB': os.path.join("models", "cyclegan_G_AB.pth"),
    'cyclegan_G_BA': os.path.join("models", "cyclegan_G_BA.pth"),
    'highres_gan_generator': os.path.join("models", "highres_gan_generator.pth")
}

# GitHub release URLs for models - pointing to the actual repository
GITHUB_RELEASE_URLS = {
    'resnet101': "https://github.com/mohdAbbas11/alzheimer-classification/raw/main/models/best_resnet101.pth",
    'resnext101_32x8d': "https://github.com/mohdAbbas11/alzheimer-classification/raw/main/models/best_resnext101_32x8d.pth",
    'densenet161': "https://github.com/mohdAbbas11/alzheimer-classification/raw/main/models/best_densenet161.pth",
    'ensemble': "https://github.com/mohdAbbas11/alzheimer-classification/raw/main/models/best_alzheimer_model.pth",
    'cgan_generator': "https://github.com/mohdAbbas11/alzheimer-classification/raw/main/models/cgan_generator.pth",
    'cyclegan_G_AB': "https://github.com/mohdAbbas11/alzheimer-classification/raw/main/models/cyclegan_G_AB.pth",
    'cyclegan_G_BA': "https://github.com/mohdAbbas11/alzheimer-classification/raw/main/models/cyclegan_G_BA.pth",
    'highres_gan_generator': "https://github.com/mohdAbbas11/alzheimer-classification/raw/main/models/highres_gan_generator.pth"
}

# Function to download model if not present
def download_model_if_needed(model_name):
    model_path = MODEL_PATHS[model_name]
    if not os.path.exists(model_path):
        st.info(f"Downloading {model_name} model... This may take a while.")
        try:
            url = GITHUB_RELEASE_URLS[model_name]
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"Downloaded {model_name} model successfully.")
            return True
        except Exception as e:
            st.error(f"Failed to download {model_name} model: {str(e)}")
            return False
    return True

# Function to safely load model state
def safe_load_state_dict(model_path, device_to_use):
    try:
        # First try normal loading
        return torch.load(model_path, map_location=device_to_use)
    except RuntimeError as e:
        if "Tried to instantiate class" in str(e) or "_get_custom_class" in str(e):
            st.warning(f"Custom class error when loading model. Trying to load with pickle_module=None.")
            # Try loading with pickle_module=None to avoid custom class issues
            try:
                return torch.load(model_path, map_location=device_to_use, pickle_module=None)
            except Exception as inner_e:
                st.error(f"Failed alternative loading: {str(inner_e)}")
                return None
        raise e

# Function to load classification models
@st.cache_resource
def load_classification_models(force_cpu=False):
    # Get device based on force_cpu setting
    device_to_use = torch.device("cpu") if force_cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    models = {}
    
    # Load individual models
    for model_name in ['resnet101', 'resnext101_32x8d', 'densenet161']:
        model_path = MODEL_PATHS[model_name]
        if not os.path.exists(model_path) and not download_model_if_needed(model_name):
            st.warning(f"Model file {model_path} not found and couldn't be downloaded.")
            continue
            
        try:
            model = EnhancedAlzheimerNet(model_name=model_name, num_classes=len(categories))
            state_dict = safe_load_state_dict(model_path, device_to_use)
            if state_dict is None:
                st.warning(f"Failed to load {model_name} model state.")
                continue
                
            model.load_state_dict(state_dict)
            model = model.to(device_to_use)
            model.eval()
            models[model_name] = model
            st.success(f"Successfully loaded {model_name} model.")
        except Exception as e:
            st.warning(f"Error loading {model_name} model: {str(e)}")
    
    # Try to load ensemble model if available
    ensemble_path = MODEL_PATHS['ensemble']
    if not os.path.exists(ensemble_path) and not download_model_if_needed('ensemble'):
        st.warning(f"Ensemble model file {ensemble_path} not found and couldn't be downloaded.")
    else:
        try:
            # Try to load the ensemble model - use resnext101_32x8d architecture instead of generic 'ensemble'
            ensemble_model = EnhancedAlzheimerNet(model_name='resnext101_32x8d', num_classes=len(categories))
            state_dict = safe_load_state_dict(ensemble_path, device_to_use)
            if state_dict is None:
                st.warning(f"Failed to load ensemble model state.")
            else:
                ensemble_model.load_state_dict(state_dict)
                ensemble_model = ensemble_model.to(device_to_use)
                ensemble_model.eval()
                models['ensemble'] = ensemble_model
                st.success("Successfully loaded ensemble model.")
        except Exception as e:
            st.warning(f"Error loading ensemble model: {str(e)}")
            st.info("Will fall back to real-time ensemble of individual models.")
    
    return models, device_to_use

# Function to load GAN models
@st.cache_resource
def load_gan_models(force_cpu=False):
    # Get device based on force_cpu setting
    device_to_use = torch.device("cpu") if force_cpu else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    models = {}
    
    # Load CGAN model
    cgan_path = MODEL_PATHS['cgan_generator']
    if not os.path.exists(cgan_path) and not download_model_if_needed('cgan_generator'):
        st.warning(f"CGAN model file {cgan_path} not found and couldn't be downloaded.")
    else:
        try:
            cgan = CGAN()
            state_dict = safe_load_state_dict(cgan_path, device_to_use)
            if state_dict is None:
                st.warning(f"Failed to load CGAN model state.")
            else:
                cgan.load_state_dict(state_dict)
                cgan = cgan.to(device_to_use)
                cgan.eval()
                models["CGAN"] = cgan
                st.success("Successfully loaded CGAN model.")
        except Exception as e:
            st.warning(f"Error loading CGAN model: {str(e)}")
    
    # Load CycleGAN models
    cyclegan_path_G_AB = MODEL_PATHS['cyclegan_G_AB']
    cyclegan_path_G_BA = MODEL_PATHS['cyclegan_G_BA']
    cycleGAN_AB_exists = os.path.exists(cyclegan_path_G_AB) or download_model_if_needed('cyclegan_G_AB')
    cycleGAN_BA_exists = os.path.exists(cyclegan_path_G_BA) or download_model_if_needed('cyclegan_G_BA')
    
    if cycleGAN_AB_exists and cycleGAN_BA_exists:
        try:
            cyclegan_G_AB = CycleGAN(direction="AB")
            cyclegan_G_BA = CycleGAN(direction="BA")
            state_dict_AB = safe_load_state_dict(cyclegan_path_G_AB, device_to_use)
            state_dict_BA = safe_load_state_dict(cyclegan_path_G_BA, device_to_use)
            
            if state_dict_AB is None or state_dict_BA is None:
                st.warning(f"Failed to load one or both CycleGAN model states.")
            else:
                cyclegan_G_AB.load_state_dict(state_dict_AB)
                cyclegan_G_BA.load_state_dict(state_dict_BA)
                cyclegan_G_AB = cyclegan_G_AB.to(device_to_use)
                cyclegan_G_BA = cyclegan_G_BA.to(device_to_use)
                cyclegan_G_AB.eval()
                cyclegan_G_BA.eval()
                models["CycleGAN_AB"] = cyclegan_G_AB
                models["CycleGAN_BA"] = cyclegan_G_BA
                st.success("Successfully loaded CycleGAN models.")
        except Exception as e:
            st.warning(f"Error loading CycleGAN models: {str(e)}")
    
    # Load HighResGAN model
    hires_gan_path = MODEL_PATHS['highres_gan_generator']
    if not os.path.exists(hires_gan_path) and not download_model_if_needed('highres_gan_generator'):
        st.warning(f"HighResGAN model file {hires_gan_path} not found and couldn't be downloaded.")
    else:
        try:
            hires_gan = HighResGAN()
            state_dict = safe_load_state_dict(hires_gan_path, device_to_use)
            if state_dict is None:
                st.warning(f"Failed to load HighResGAN model state.")
            else:
                hires_gan.load_state_dict(state_dict)
                hires_gan = hires_gan.to(device_to_use)
                hires_gan.eval()
                models["HighResGAN"] = hires_gan
                st.success("Successfully loaded HighResGAN model.")
        except Exception as e:
            st.warning(f"Error loading HighResGAN model: {str(e)}")
    
    if not models:
        st.warning("No GAN models found or loaded successfully.")
        st.info("Check that the model files exist in the 'models' directory.")
        st.info("Required GAN model files: cgan_generator.pth, cyclegan_G_AB.pth, cyclegan_G_BA.pth, highres_gan_generator.pth")
    
    return models, device_to_use

# Function to preprocess image for classification
def preprocess_image(image, device_to_use):
    transform = data_transforms['test']
    image = transform(image).unsqueeze(0).to(device_to_use)
    return image

# Function to classify image using a single model
def classify_image(model, image):
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        
    return predicted.item(), {categories[i]: float(probabilities[i]) for i in range(len(categories))}

# Function to classify image using ensemble of models
def ensemble_classify(models, image):
    # Collect outputs from all models
    all_outputs = []
    
    with torch.no_grad():
        for model_name, model in models.items():
            if model_name != 'ensemble':  # Skip the ensemble model itself
                outputs = model(image)
                all_outputs.append(outputs)
        
        # Average the logits
        if all_outputs:
            ensemble_output = torch.mean(torch.stack(all_outputs), dim=0)
            probabilities = torch.nn.functional.softmax(ensemble_output, dim=1)[0]
            _, predicted = torch.max(ensemble_output, 1)
            
            return predicted.item(), {categories[i]: float(probabilities[i]) for i in range(len(categories))}
        else:
            # Fallback to ensemble model if individual models aren't available
            return classify_image(models['ensemble'], image)

# Function to generate images using GANs
def generate_images(gan_model, gan_type, device_to_use, condition=None, input_image=None):
    with torch.no_grad():
        if gan_type == "CGAN":
            # For CGAN, we need a condition (class label)
            noise = torch.randn(1, 100).to(device_to_use)
            label = torch.tensor([condition]).to(device_to_use)
            generated = gan_model(noise, label)
        
        elif "CycleGAN" in gan_type:
            # For CycleGAN, we need an input image
            if input_image is None:
                return None
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            input_tensor = transform(input_image).unsqueeze(0).to(device_to_use)
            generated = gan_model(input_tensor)
        
        elif gan_type == "HighResGAN":
            # For HighResGAN, we can use random noise or condition on a class
            noise = torch.randn(1, 128).to(device_to_use)
            if condition is not None:
                label = torch.tensor([condition]).to(device_to_use)
                generated = gan_model(noise, label)
            else:
                generated = gan_model(noise)
    
    # Convert to image
    generated = generated.cpu().squeeze(0)
    generated = (generated + 1) / 2.0  # Denormalize from [-1,1] to [0,1]
    generated_np = generated.permute(1, 2, 0).numpy()
    generated_np = np.clip(generated_np, 0, 1)
    
    return generated_np

# Main app
def main():
    st.title("🧠 Alzheimer's Disease Detection & Image Generation")
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose the app mode", ["Classification", "Image Generation", "About"])
    
    # Add device selection option
    st.sidebar.title("Device Settings")
    force_cpu = st.sidebar.checkbox("Force CPU Usage (prevents CUDA out of memory errors)", value=True)
    
    if force_cpu:
        st.sidebar.info("Using CPU for computation")
        # Set environment variable for other scripts
        os.environ['FORCE_CPU'] = '1'
    else:
        if torch.cuda.is_available():
            st.sidebar.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            os.environ['FORCE_CPU'] = '0'
        else:
            st.sidebar.info("No GPU detected, using CPU")
            os.environ['FORCE_CPU'] = '1'
    
    # Load models with force_cpu parameter - with error handling
    try:
        with st.spinner("Loading classification models..."):
            classification_models, device_to_use = load_classification_models(force_cpu=force_cpu)
        if not classification_models:
            st.warning("No classification models were loaded successfully.")
            
        with st.spinner("Loading GAN models..."):
            gan_models, device_to_use = load_gan_models(force_cpu=force_cpu)
        if not gan_models:
            st.warning("No GAN models were loaded successfully.")
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        classification_models = {}
        gan_models = {}
        device_to_use = torch.device("cpu")
    
    if app_mode == "Classification":
        st.header("Alzheimer's Disease Classification")
        st.write("Upload a brain MRI scan to classify the stage of Alzheimer's disease.")
        
        # Model selection
        model_options = list(classification_models.keys())
        
        if model_options:
            # Add ensemble option if we have at least 2 models
            if len(model_options) >= 2 and 'ensemble' not in model_options:
                model_options.append('ensemble')
                
            selected_model = st.selectbox("Select Classification Model", 
                                         options=model_options, 
                                         index=model_options.index('ensemble') if 'ensemble' in model_options else 0,
                                         format_func=lambda x: f"{x.capitalize()} Model")
            
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Save the uploaded image
                save_path = os.path.join("uploaded_images", uploaded_file.name)
                image.save(save_path)
                
                # Preprocess the image
                processed_image = preprocess_image(image, device_to_use)
                
                # Classify the image
                with st.spinner("Classifying..."):
                    if selected_model == 'ensemble' and len(classification_models) > 1:
                        class_idx, probabilities = ensemble_classify(classification_models, processed_image)
                        st.info("Using real-time ensemble of all available models for classification.")
                    else:
                        class_idx, probabilities = classify_image(classification_models[selected_model], processed_image)
                
                with col2:
                    st.subheader("Classification Results")
                    st.write(f"**Predicted Class:** {categories[class_idx]}")
                    
                    # Display probabilities as a bar chart
                    fig, ax = plt.subplots()
                    categories_list = list(probabilities.keys())
                    values = list(probabilities.values())
                    ax.barh(categories_list, values)
                    ax.set_xlim(0, 1)
                    ax.set_xlabel('Probability')
                    ax.set_title('Classification Probabilities')
                    st.pyplot(fig)
                    
                    # If using ensemble, show individual model predictions
                    if selected_model == 'ensemble' and len(classification_models) > 1:
                        st.subheader("Individual Model Predictions")
                        
                        for model_name, model in classification_models.items():
                            if model_name != 'ensemble':  # Skip showing ensemble results again
                                ind_class_idx, ind_probs = classify_image(model, processed_image)
                                st.write(f"**{model_name.capitalize()}:** {categories[ind_class_idx]}")
        else:
            st.warning("No classification models found. Please ensure the model files are in the correct location.")
            st.info("Check that the model files exist in the 'models' directory.")
            st.info("Required files: best_resnet101.pth, best_resnext101_32x8d.pth, best_densenet161.pth")
    
    elif app_mode == "Image Generation":
        st.header("Brain MRI Image Generation")
        st.write("Generate synthetic brain MRI images using different GAN models.")
        
        # Check if models are available
        if not gan_models:
            st.warning("No GAN models were loaded successfully.")
            st.info("Check that the model files exist in the 'models' directory.")
            st.info("Required GAN model files: cgan_generator.pth, cyclegan_G_AB.pth, cyclegan_G_BA.pth, highres_gan_generator.pth")
            return
        
        # Select GAN model
        available_models = list(gan_models.keys())
        st.success(f"Found {len(available_models)} GAN model(s): {', '.join(available_models)}")
        gan_type = st.selectbox("Select GAN Model", available_models)
        
        if gan_type == "CGAN":
            st.subheader("Conditional GAN (CGAN)")
            st.write("Generate brain MRI images conditioned on Alzheimer's stage.")
            
            # Select condition (class)
            condition = st.selectbox("Select Alzheimer's Stage", 
                                    list(range(len(categories))), 
                                    format_func=lambda x: categories[x])
            
            if st.button("Generate Image"):
                with st.spinner("Generating image..."):
                    try:
                        generated_image = generate_images(gan_models[gan_type], gan_type, device_to_use, condition=condition)
                        
                        if generated_image is not None:
                            st.image(generated_image, caption=f"Generated {categories[condition]} MRI", 
                                    clamp=True, use_column_width=True)
                            
                            # Save generated image
                            filename = f"cgan_{categories[condition]}_{np.random.randint(1000)}.png"
                            save_path = os.path.join("generated_images", filename)
                            plt.imsave(save_path, generated_image)
                            st.success(f"Image saved to {save_path}")
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")
        
        elif "CycleGAN" in gan_type:
            direction = "AB" if gan_type == "CycleGAN_AB" else "BA"
            source = "Normal" if direction == "AB" else "Alzheimer's"
            target = "Alzheimer's" if direction == "AB" else "Normal"
            
            st.subheader(f"CycleGAN: {source} to {target}")
            st.write(f"Transform {source} brain MRI images to {target} brain MRI images.")
            
            uploaded_file = st.file_uploader(f"Upload a {source} brain MRI image...", 
                                           type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                input_image = Image.open(uploaded_file).convert('RGB')
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.image(input_image, caption=f"Input {source} Image", use_column_width=True)
                
                if st.button("Generate Image"):
                    with st.spinner("Generating image..."):
                        try:
                            generated_image = generate_images(gan_models[gan_type], gan_type, device_to_use, 
                                                         input_image=input_image)
                            
                            if generated_image is not None:
                                with col2:
                                    st.image(generated_image, caption=f"Generated {target} Image", 
                                            clamp=True, use_column_width=True)
                                    
                                    # Save generated image
                                    filename = f"cyclegan_{source}_to_{target}_{np.random.randint(1000)}.png"
                                    save_path = os.path.join("generated_images", filename)
                                    plt.imsave(save_path, generated_image)
                                    st.success(f"Image saved to {save_path}")
                        except Exception as e:
                            st.error(f"Error generating image: {str(e)}")
        
        elif gan_type == "HighResGAN":
            st.subheader("High-Resolution GAN")
            st.write("Generate high-resolution brain MRI images.")
            
            # Option to condition on class
            use_condition = st.checkbox("Condition on Alzheimer's Stage")
            
            condition = None
            if use_condition:
                condition = st.selectbox("Select Alzheimer's Stage", 
                                        list(range(len(categories))), 
                                        format_func=lambda x: categories[x])
            
            if st.button("Generate Image"):
                with st.spinner("Generating high-resolution image..."):
                    try:
                        generated_image = generate_images(gan_models[gan_type], gan_type, device_to_use, condition=condition)
                        
                        if generated_image is not None:
                            caption = f"Generated High-Res MRI"
                            if condition is not None:
                                caption += f" ({categories[condition]})"
                                
                            st.image(generated_image, caption=caption, clamp=True, use_column_width=True)
                            
                            # Save generated image
                            condition_str = f"_{categories[condition]}" if condition is not None else ""
                            filename = f"highres{condition_str}_{np.random.randint(1000)}.png"
                            save_path = os.path.join("generated_images", filename)
                            plt.imsave(save_path, generated_image)
                            st.success(f"Image saved to {save_path}")
                    except Exception as e:
                        st.error(f"Error generating image: {str(e)}")
    
    else:  # About
        st.header("About")
        st.write("""
        ## Alzheimer's Disease Detection & Image Generation
        
        This application provides tools for:
        
        1. **Classification**: Upload brain MRI scans to detect the stage of Alzheimer's disease.
           - **ResNet101**: Deep residual network with 101 layers
           - **ResNeXt101_32x8d**: Residual network with grouped convolutions
           - **DenseNet161**: Densely connected convolutional network
           - **Ensemble**: Combines predictions from all models for better accuracy
        
        2. **Image Generation**: Generate synthetic brain MRI images using various GAN models:
           - **CGAN (Conditional GAN)**: Generate images conditioned on Alzheimer's stage.
           - **CycleGAN**: Transform between normal and Alzheimer's affected brain images.
           - **High-Resolution GAN**: Generate detailed, high-resolution brain MRI images.
        
        ### Alzheimer's Disease Stages
        - **NonDemented**: Normal cognitive function
        - **VeryMildDemented**: Very mild cognitive decline
        - **MildDemented**: Mild cognitive decline
        - **ModerateDemented**: Moderate cognitive decline
        
        ### Dataset
        The models were trained on segmented brain MRI images from the OASIS and ADNI datasets.
        
        ### Models
        - Classification: ResNet101, ResNeXt101, DenseNet161 ensemble
        - GANs: CGAN, CycleGAN, HighResGAN
        """)

if __name__ == "__main__":
    try:
        # Display environment information
        st.sidebar.markdown("### Environment Info")
        st.sidebar.info(f"Python version: {sys.version}")
        st.sidebar.info(f"PyTorch version: {torch.__version__}")
        st.sidebar.info(f"Device: {torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')}")
        
        # Add option to skip model loading for debugging
        skip_models = st.sidebar.checkbox("Skip model loading (debug mode)", value=False)
        
        if skip_models:
            st.warning("Running in debug mode - model loading skipped")
            # Run a simplified version of the app
            st.title("🧠 Alzheimer's Disease Detection & Image Generation (Debug Mode)")
            st.write("Application is running in debug mode. Model loading is skipped.")
            st.info("To use the full application, uncheck 'Skip model loading' in the sidebar.")
        else:
            # Normal app execution
            main()
    except Exception as e:
        st.error(f"❌ Application crashed: {str(e)}")
        st.error(f"Exception type: {type(e).__name__}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        st.warning("Try enabling 'Skip model loading' in the sidebar to run in debug mode.") 