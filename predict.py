import torch
import torch.nn.functional as F
import src.config as config
from src.model import CatDogCNN
from torchvision import transforms
from PIL import Image

def predict_image(image_path, model_path):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CatDogCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

        probabilties = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilties, 1)

        classes = ['Cat', 'Dog']
        result = classes[predicted.item()]
        conf_score = confidence.item() * 100

        print(f"Result: {result} ({conf_score:.2f}%)")
        return result, conf_score
        

if __name__ == "__main__":
    test_img = "images\\dog.jpg"
    print(predict_image(test_img, config.MODEL_PATH))