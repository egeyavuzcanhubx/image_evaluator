
from PIL import Image
from image_evaluator import ImageEvaluator  

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
    #-- If you want to directly pass the image path:
    # results = evaluator.evaluate(img_path, prompt)
    
    
    print("ViLT Similarity:", results["vilt_similarity"])
    print("Quality Score:", results["quality_score"])
    print("Aesthetic Score:", results["aesthetic_score"])

if __name__ == "__main__":
    main()
