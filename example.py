# example.py
from artdetectors import ImageAnalysisPipeline

# Use transfer learning (3 classes)
pipe = ImageAnalysisPipeline(
    use_transfer_learning=True
)

test_img = "example_images/dalle/dalle_input2.jpeg"
result_img = "example_results/dalle_test2.png"

# 1) Fast detectors only
det = pipe.analyze(test_img)
print("\n=== DETECTORS ===")
print("styles:", det["styles"])
print("caption:", det["caption"])
print("susy:", det["susy"])
# susy will now have 3 classes: authentic, midjourney, dalle3

# 2) Slow restyling
out = pipe.restyle_image(
    test_img,
    target_style="impressionism",
    strength=0.5,
    guidance_scale=5.0,
    num_inference_steps=18,
)
out["restyled_image"].save(result_img)
print(f"\nSaved new image: {result_img}")