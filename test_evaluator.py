# test_image_evaluator.py

from PIL import Image
from image_evaluator import ImageEvaluator  # Adjust the import path if necessary

def main():
    evaluator = ImageEvaluator()
    
    # Load the image as a PIL Image
    img_path = "/home/egeyavuzcan/metric-flux/dimg-181.png"
    
    #image = Image.open(image_path).convert("RGB")
    image = evaluator._load_image(img_path)
    
    # Define the prompt
    prompt = "A woman "
    
    # Evaluate the image and prompt
    results = evaluator.evaluate(image, prompt)
    #--> If you have a PIL image instead of a file path:
    #image = Image.open("/home/egeyavuzcan/car/1.png").convert("RGB")
    #results = evaluator.evaluate(image, prompt)
    
    print("ViLT Similarity:", results["vilt_similarity"])
    print("Quality Score:", results["quality_score"])
    print("Aesthetic Score:", results["aesthetic_score"])

if __name__ == "__main__":
    main()
