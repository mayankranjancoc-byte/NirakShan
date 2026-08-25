import os
import sys
import cv2
import json
import tempfile

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend')))

from modules.ocr_extraction import extract_document_fields

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    original_img_path = os.path.abspath(
        os.path.join(base_dir, "backend", "vendor", "fastmrz", "data", "passport_uk.jpg")
    )
    
    print(f"Testing original image: {original_img_path}")
    res_orig = extract_document_fields(original_img_path)
    print("Original result status:", res_orig.get("status"))
    
    # Read the image
    img = cv2.imread(original_img_path)
    if img is None:
        print("Failed to read image")
        return
        
    temp_dir = tempfile.gettempdir()
    
    # Test 1: Heavy Blur
    print("\n--- Testing Blur ---")
    blurry_img = cv2.GaussianBlur(img, (9, 9), 0)
    blur_path = os.path.join(temp_dir, "blur_passport.jpg")
    cv2.imwrite(blur_path, blurry_img)
    
    res_blur = extract_document_fields(blur_path)
    print("Blur result status:", res_blur.get("status"))
    print("Is blurry according to IQA?", res_blur.get("iqa_metrics", {}).get("is_blurry"))
    
    # Test 2: Low contrast (simulating shadow)
    print("\n--- Testing Shadow/Low Contrast ---")
    shadow_img = cv2.convertScaleAbs(img, alpha=0.5, beta=10)
    shadow_path = os.path.join(temp_dir, "shadow_passport.jpg")
    cv2.imwrite(shadow_path, shadow_img)
    
    res_shadow = extract_document_fields(shadow_path)
    print("Shadow result status:", res_shadow.get("status"))
    if res_shadow.get("is_corrected_by_disambiguation"):
        print("-> Fixed by Checksum Disambiguation!")

if __name__ == "__main__":
    main()
