import os
from PIL import Image
import torch
from torchvision import transforms, models, datasets
import pyiqa
from transformers import ViltProcessor, ViltForImageAndTextRetrieval

class ImageEvaluator:
    def __init__(self, device: str = None):
        
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize ViLT processor and model
        self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        self.model_vilt = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        self.model_vilt.to(self.device)
        self.model_vilt.eval()
        
        # Initialize quality and aesthetic metric (using QALIGN)
        self.qalign = pyiqa.create_metric('qalign').to(self.device)
        
    def _load_image(self, image_input):
        """
        Accepts either a file path or a PIL Image and returns a PIL Image.
        """
        if isinstance(image_input, Image.Image):
            return image_input
        elif isinstance(image_input, str) and os.path.exists(image_input):
            return Image.open(image_input).convert("RGB")
        else:
            raise ValueError("Input must be a valid image path or a PIL Image.")
    
    def _calculate_quality_score(self, image: Image.Image) -> float:
        """
        Computes quality score using QALIGN with a PIL image.
        """
        return self.qalign(image, task_='quality').item()

    def _calculate_aesthetic_score(self, image: Image.Image) -> float:
        """
        Computes aesthetic score using QALIGN with a PIL image.
        """
        return self.qalign(image, task_='aesthetic').item()

    def evaluate(self, image_input, prompt: str) -> dict:
        """
        Given an image (PIL image or file path) and a text prompt, returns:
        - vilt_similarity: ViLT similarity score between the image and prompt
        - quality_score: Quality score computed by QALIGN
        - aesthetic_score: Aesthetic score computed by QALIGN
        """
        # Load the image
        image = self._load_image(image_input)
        
        # Process image and prompt using ViLT processor
        encoding = self.processor(images=image, text=prompt, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model_vilt(**encoding)
        
        # ViLT returns a similarity score from logits
        vilt_similarity = outputs.logits[0, :].item()
        quality_score = self._calculate_quality_score(image)
        aesthetic_score = self._calculate_aesthetic_score(image)
        
        return {
            "vilt_similarity": vilt_similarity,
            "quality_score": quality_score,
            "aesthetic_score": aesthetic_score
        }


# Example usage:
if __name__ == "__main__":
    evaluator = ImageEvaluator()
    image_path = "/home/egeyavuzcan/car/1.png" 
    prompt = " A blue muscle car driving down a rural road at sunset"
    
    results = evaluator.evaluate(image_path, prompt)
    
    # --> If you have a PIL image instead of a file path:
    #image = Image.open("/home/egeyavuzcan/car/1.png").convert("RGB")
    #results = evaluator.evaluate(image, prompt)
    
    print("ViLT Similarity:", results["vilt_similarity"])
    print("Quality Score:", results["quality_score"])
    print("Aesthetic Score:", results["aesthetic_score"])
