
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image,  ImageDraw, ImageFont
import io, os,cv2,  uuid
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, TypedDict


from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='en'
)

DETECTED_CAR_FOLDER = "detected_cars"
CONF_THRESHOLD = 0.75
model_licence = YOLO("best.pt")
model = YOLO("yolo11l.pt")
app = FastAPI(title='Car Detection AI API', version=1.0)

 

@app.post('/test')
async def test_endpoint(image: UploadFile = File(...)):
    try: 
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
       

    except Exception:
        raise HTTPException(400, "Invalid image") 
    results = model_licence(img)
    detections = []

    print(f"type of result: {type(results)}")
    # Draw rectangles on the image for each detection
    img_copy = img.copy()
    
    predicted_areas = []
    parsed = []
    for r in results:
        print(f"type of r: {type(r)}")
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0].item()) if hasattr(box, 'cls') else None
                confidence = float(box.conf[0].item()) if hasattr(box, 'conf') else None
                bbox = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else None
                detections.append({
                    'class_id': cls_id,
                    'confidence': confidence,
                    'bbox': bbox
                })
                # Draw rectangle if bbox is valid
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                   
                    # Save cropped predicted area
                    crop = img_copy.crop((x1, y1, x2, y2))
                    crop_path = f"detected_cars/predicted_{uuid.uuid4().hex[:8]}.jpg"
                    crop.save(crop_path)
                    predicted_areas.append(crop_path)
                    crop_np = np.array(crop) 
                    bigger = cv2.resize(crop_np, None, fx=20, fy=20)
                    gray = cv2.cvtColor(bigger, cv2.COLOR_BGR2GRAY)
                    blur = cv2.GaussianBlur(gray, (5,5), 0)
                    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
                    processed_crop = Image.fromarray(thresh)
                    processed_crop.save(f"detected_cars/predicted_filter_{uuid.uuid4().hex[:8]}.jpg")

                    if len(thresh.shape) == 2:  # If grayscale
                        thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
                    else:
                        thresh_rgb = thresh
                    result = ocr.predict(thresh_rgb)
                    
                    for res in result:
                        print(f"res: {res['rec_texts']}, score: {res['rec_scores']}")
                        
                        if (res['rec_texts'] and res['rec_texts'][0] and len( "".join(res['rec_texts'])) >=4):
                            parsed.append({
                                "text": "".join(res['rec_texts']).replace(" ", ""),
                                "score": sum(res['rec_scores']) / len(res['rec_scores']) if res['rec_scores'] else 0,
                            })
                    

    return {"detections": detections,  "predicted_areas": predicted_areas, "ocr_results": parsed}


@app.post("/detect")
async def detect_cars(image: UploadFile = File(...)):
    """
    Basic car detection endpoint - fast detection only
    Returns car count and basic bounding box information
    """
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Invalid image")

    # Detect cars in image
    detected_cars = detect_cars_in_image(img)
    
    if not detected_cars:
        raise HTTPException(404, "No cars detected")

    license_plates = detect_license_plate(detected_cars)
    if not license_plates:
        raise HTTPException(404, "No license plates detected")
    return {"license_plates": license_plates}

list_of_cars = [2,5,7]

class CarInfo(TypedDict):
    car_id: str
    confidence: float
    bbox: Dict[str, int]
    image_file: str
    car_crop: Image

def detect_cars_in_image(image: Image.Image) -> list[CarInfo]:
    results = model(image)

    detected_cars = []
    img_width, img_height = image.size

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
            
        for box in boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            # Filter for objects in list_of_cars with sufficient confidence
            if cls_id not in list_of_cars or confidence < CONF_THRESHOLD:
                continue

            # Get bounding box coordinates
            car_x1, car_y1, car_x2, car_y2 = map(int, box.xyxy[0].tolist())
            
            # Ensure coordinates are within image bounds
            car_x1, car_y1 = max(0, car_x1), max(0, car_y1)
            car_x2, car_y2 = min(img_width, car_x2), min(img_height, car_y2)
            
            # Skip if bounding box is invalid
            if car_x2 <= car_x1 or car_y2 <= car_y1:
                continue
            
            car_id = f"car_{uuid.uuid4().hex[:8]}"
            
            # Crop and save car image
            car_crop = image.crop((car_x1, car_y1, car_x2, car_y2))
            # Basic car info
            car_info = {
                "car_id": car_id,
                "confidence": round(confidence, 3),
                "bbox": {
                    "x1": car_x1, 
                    "y1": car_y1, 
                    "x2": car_x2, 
                    "y2": car_y2,
                    "width": car_x2 - car_x1,
                    "height": car_y2 - car_y1
                },
                # "image_file": filepath,
                "car_crop": car_crop
            }
            detected_cars.append(car_info)

    return detected_cars

def detect_license_plate(cars_list: List[CarInfo]) -> List:
    parsed = []
    for car in cars_list:
        license_plate = model_licence(car["car_crop"])
        for r in license_plate:
            if hasattr(r, 'boxes') and r.boxes is not None:
                for box in r.boxes:
                    
                    bbox = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else None
                    
                    # Draw rectangle if bbox is valid
                    if bbox and len(bbox) == 4:
                        x1, y1, x2, y2 = map(int, bbox)
                    
                        # Save cropped predicted area
                        crop = car["car_crop"].crop((x1, y1, x2, y2))
                        
                        
                        crop_np = np.array(crop) 
                        bigger = cv2.resize(crop_np, None, fx=20, fy=20)
                        gray = cv2.cvtColor(bigger, cv2.COLOR_BGR2GRAY)
                        blur = cv2.GaussianBlur(gray, (5,5), 0)
                        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
                        

                        if len(thresh.shape) == 2:  # If grayscale
                            thresh_rgb = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
                        else:
                            thresh_rgb = thresh
                        result = ocr.predict(crop_np)
                      
                        for res in result:
                            
                            if (res['rec_texts'] and res['rec_texts'][0]):
                                parsed_result = parse_text(res)
                                if( len("".join(parsed_result['rec_texts']).replace(" ", "")) == 8):
                                    parsed.append({
                                        "car_id": car["car_id"],
                                        "car_confidence": car["confidence"],
                                        "car_bbox": car["bbox"],
                                        "text": correct_text(join_text_left_to_right(parsed_result)),
                                        "score": sum(res['rec_scores']) / len(res['rec_scores']) if res['rec_scores'] else 0,
                                    })
    return parsed

def parse_text(result):
    # filtered_list = [s for s in result if s.lower() != 'ua']
    # return filtered_list
    rec_texts = []
    rec_boxes = []
    for idx, value in enumerate(result['rec_texts']):
        if value.lower() != 'ua':
            rec_texts.append(value)
            rec_boxes.append(result['rec_boxes'][idx])
    return {"rec_texts": rec_texts, "rec_boxes": rec_boxes}


def join_text_left_to_right(result):
    if isinstance(result, dict) and 'rec_boxes' in result and 'rec_texts' in result:
        pairs = []
        for text, box in zip(result['rec_texts'], result['rec_boxes']):
            left_x = min(box[0], box[2])
            pairs.append((left_x, text))
        pairs.sort(key=lambda x: x[0])
        return "".join([p[1] for p in pairs]).replace(" ", "")
    elif isinstance(result, dict):
        # Fallback for dict
        return "".join(result.get('rec_texts', [])).replace(" ", "")
    elif isinstance(result, list):
        # Fallback for list
        return "".join(result).replace(" ", "")
    else:
        return ""


def correct_text(text: str) -> str:
    # Convert to list for mutability
    chars = list(text)
    # First or second position
    for i in [0, 1]:
        if i < len(chars):
            if chars[i] == '1':
                chars[i] = 'I'
            elif chars[i] == '0':
                chars[i] = 'O'
                 # Middle positions (2 to 5)
    for i in range(2, 6):
        if i < len(chars):
            if chars[i] == 'I':
                chars[i] = '1'
            elif chars[i] == 'O':
                chars[i] = '0'
    # Last two positions
    for i in [-2, -1]:
        if abs(i) <= len(chars):
            if chars[i] == '1':
                chars[i] = 'I'
            elif chars[i] == '0':
                chars[i] = 'O'
    return ''.join(chars)