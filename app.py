import gradio as gr
from PIL import Image
import torch
from src.model import CatDogCNN
import src.config as config
from torchvision import transforms
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CatDogCNN().to(device)
model.load_state_dict(torch.load(config.MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
])

def classify_image(input_img):
    if input_img is None:
        return None
    
    img = Image.fromarray(input_img.astype('uint8'), 'RGB')
    image_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

        categories = ['Cat', 'Dog']
        return {categories[i]: float(probabilities[i]) for i in range(len(categories))}
    
# App's interface
demo = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=2),
    title="🐱 Cats and Dogs Classifier 🐶",
    description="Upload a cat/dog photo — the neural network will identify it.",
    examples=["images/cat_2.jpg", "images/dog.jpg"]
)

if __name__ == "__main__":
    demo.launch(share=True)