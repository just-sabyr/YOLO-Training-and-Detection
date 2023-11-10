### Running inference on images
### Function annotate() can be imported and used in another file
### if this code is run on it's own, it labels objects in all images in folder 'images'

import os

from ultralytics import YOLO
import cv2

# downloads yolov8 model if is not downloaded yet then loads the model
model = YOLO('runs/detect/train3/weights/best.pt')

def annotate(img_path, show=False):    
    """
    :params: img_path: path to an image
    :show: Boolean: whether to show the annotated image or not
    :returns: annotated image
    """

    results = model(img_path)

    annotated_image = results[0].plot()    

    if show:

        # Resize for better display
        if annotated_image.shape[0] > 600 and annotated_image.shape[1] > 600:
            image_resized = cv2.resize(annotated_image, (annotated_image.shape[0]//2, annotated_image.shape[1]//2)) 
        else:
            image_resized = annotated_image

        cv2.imshow(f'YOLOv8 Inference on {img_path}', image_resized)
        cv2.waitKey(0)

    return annotated_image
    

if __name__ == '__main__':
    imgs = os.listdir('images')
    for img in imgs:
        annotated_image = annotate('images/' + img)
        cv2.imwrite('inference_images/' + img, annotated_image)



