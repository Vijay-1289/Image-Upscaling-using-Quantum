import cv2, numpy as np

def binarize_image_based_on_avg(img_gray):
    
    min_val = np.min(img_gray)
    max_val = np.max(img_gray)
    avg_val = int(np.mean(img_gray))

    print(f"Image size: {img_gray.shape}")
    print(f"Min pixel value: {min_val}")
    print(f"Max pixel value: {max_val}")
    print(f"Average pixel value: {avg_val}")

    binarized = np.where(img_gray >= avg_val, max_val, min_val).astype(np.uint8)
    return binarized

def show_side_by_side(original, binarized):
    
    combined = np.hstack((original, binarized))
    cv2.imshow('Original (Left) vs Binarized (Right)', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    image_path = 'lena_grey.jpeg'  
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img_gray is None:
        print("Failed to load image. Check the file path.")

    binarized_img = binarize_image_based_on_avg(img_gray)
    show_side_by_side(img_gray, binarized_img)