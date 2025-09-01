# # from fastapi import FastAPI, UploadFile, File, HTTPException
# # from fastapi.responses import FileResponse
# # from fastapi import Query
# # import os
# # from PIL import Image
# # import io

# # # Import configuration
# # from config import API_TITLE, API_VERSION, create_folders

# # # Import services
# # from services.car_detection import detect_cars_in_image, analyze_most_confident_car, create_car_analysis_folder
# # from services.plate_detection import detect_license_plates_in_image
# # from utils.file_management import get_image_by_id

# # # Initialize FastAPI app
# # app = FastAPI(title=API_TITLE, version=API_VERSION)

# # # Create folders on startup
# # create_folders()

# # @app.get("/")
# # async def root():
# #     """API information endpoint"""
# #     return {
# #         "message": API_TITLE, 
# #         "version": API_VERSION,
# #         "endpoints": {
# #             "detect": "POST /detect - Upload image for fast car detection",
# #             "get_details": "POST /get-details - Upload image for detailed car analysis",
# #             "detect_license_plates": "POST /detect-license-plates - Upload image for license plate detection"
# #         }
# #     }


# # @app.post("/detect")
# # async def detect_cars(image: UploadFile = File(...)):
# #     """
# #     Basic car detection endpoint - fast detection only
# #     Returns car count and basic bounding box information
# #     """
# #     try:
# #         contents = await image.read()
# #         img = Image.open(io.BytesIO(contents)).convert("RGB")
# #     except Exception:
# #         raise HTTPException(400, "Invalid image")

# #     # Detect cars in image
# #     detected_cars = detect_cars_in_image(img)
    
# #     if not detected_cars:
# #         raise HTTPException(404, "No cars detected")

# #     # Prepare download URL for outlined image
# #     outlined_image_file = detected_cars['outlined_image_file']
# #     if outlined_image_file:
# #         # Only return the filename, not the full path, for security
# #         filename = os.path.basename(outlined_image_file)
# #         download_url = f"/download-best-car-image?file={filename}"
# #     else:
# #         download_url = None

# #     return {
# #         "cars_detected": detected_cars["detected_cars_count"],
# #         "cars": detected_cars['detected_cars'],
# #         "best_car": detected_cars['best_car'],
# #         "best_car_image_download_url": download_url
# #     }

# # # Endpoint to download the best car image by filename
# # @app.get("/download-best-car-image")
# # async def download_best_car_image(file: str = Query(..., description="Filename of the best car image to download")):
# #     # Build the full file path
# #     from config import DELETCTED_CAR_FOLDER
# #     file_path = os.path.join(DELETCTED_CAR_FOLDER, file)
# #     if not os.path.isfile(file_path):
# #         raise HTTPException(404, "File not found")
# #     return FileResponse(file_path, media_type="image/jpeg", filename=file)

# # @app.post("/get-details")
# # async def get_car_details(image: UploadFile = File(...)):
# #     """
# #     Get detailed analysis of the most confident car detection
# #     Creates a new folder with original image and outlined version showing details
# #     """
# #     try:
# #         contents = await image.read()
# #         img = Image.open(io.BytesIO(contents)).convert("RGB")
# #     except Exception:
# #         raise HTTPException(400, "Invalid image")
    
# #     # Analyze the most confident car
# #     car_analysis = analyze_most_confident_car(img)
    
# #     if not car_analysis:
# #         raise HTTPException(404, "No cars detected in the image")
    
# #     # Create analysis folder with files
# #     folder_info = create_car_analysis_folder(img, car_analysis)
    
# #     # Return only color information as requested
# #     return {
# #         "color": car_analysis["details"]["color"]
# #     }

# # @app.post("/detect-license-plates")
# # async def detect_license_plates(image: UploadFile = File(...)):
# #     """
# #     Detect and analyze license plates in uploaded image
# #     Returns best license plate found with debug information
# #     """
# #     try:
# #         contents = await image.read()
# #         img = Image.open(io.BytesIO(contents)).convert("RGB")
# #     except Exception:
# #         raise HTTPException(400, "Invalid image")
    
# #     # Detect license plates
# #     result = detect_license_plates_in_image(img)
    
# #     if not result["success"]:
# #         # Return debug information for failed detection
# #         return result
    
# #     # Return successful detection results
# #     return {
# #         "best_plate": result["best_plate"],
# #         "all_plates": result["all_plates"],
# #         "total_found": result["total_found"],
# #         "folder_path": result["folder_path"],
# #         "saved_files": result["saved_files"],
# #         "debug_info": result["debug_info"]
# #     }

# # @app.get("/detect-license-plate-by-id")
# # async def detect_license_plate_by_id(image_id: str = Query(..., description="Unique ID of the image to process")):
# #     # Retrieve the image using the unique ID
# #     image = get_image_by_id(image_id)
# #     if image is None:
# #         raise HTTPException(404, "Image not found")

# #     # Process the image to detect license plates
# #     result = detect_license_plates_in_image(image)

# #     if not result["success"]:
# #         # Return debug information for failed detection
# #         return result

# #     # Return successful detection results
# #     return {
# #         "best_plate": result["best_plate"],
# #         "all_plates": result["all_plates"],
# #         "total_found": result["total_found"],
# #         "folder_path": result["folder_path"],
# #         "saved_files": result["saved_files"],
# #         "debug_info": result["debug_info"]
# #     }



# from fastapi import FastAPI, UploadFile, File, HTTPException
# from PIL import Image, ImageOps, ImageDraw, ImageFont
# import io, os,cv2, easyocr, json, uuid
# import numpy as np
# from ultralytics import YOLO




# DETECTED_CAR_FOLDER = "detected_cars"
# CONF_THRESHOLD = 0.75
# model = YOLO("yolo11l.pt")
# app = FastAPI(title='Car Detection AI API', version=1.0)
# reader = easyocr.Reader(['en'])


# @app.post('/test')
# async def test_endpoint(image: UploadFile = File(...)):
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")
      
#     except Exception:
#         raise HTTPException(400, "Invalid image") 
#     results = model(img)
#     # Format results to JSON
#     detections = []
#     for r in results:
#         if hasattr(r, 'boxes') and r.boxes is not None:
#             for box in r.boxes:
#                 cls_id = int(box.cls[0].item()) if hasattr(box, 'cls') else None
#                 confidence = float(box.conf[0].item()) if hasattr(box, 'conf') else None
#                 bbox = box.xyxy[0].tolist() if hasattr(box, 'xyxy') else None
#                 detections.append({
#                     'class_id': cls_id,
#                     'confidence': confidence,
#                     'bbox': bbox
#                 })
#     return {"detections": detections}

# @app.post("/detect-license-plates")
# async def detect_license_plates(image: UploadFile = File(...)):
#     try:
#         contents = await image.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")
#         negative_image = ImageOps.invert(img)
      
#     except Exception:
#         raise HTTPException(400, "Invalid image")
    
 
#     result = []
#     for car in detect_cars_in_image(img):
#         print(car)
#         best_result = None
#         car_image = Image.open(car["image_file"]).convert("RGB")
#         negative_car_image = ImageOps.invert(car_image)
#         all_text = reader.readtext(np.array(negative_car_image))
#         negative_car_image.save(f"{car['car_id']}_output.jpg")
#         print('all_text:', all_text)
#         for (bbox, text, confidence) in all_text:
#             if confidence > 0.5:
#                 if best_result is None or confidence > best_result[2]:
#                     best_result = (bbox, text, confidence)
       

#     all_reads = reader.readtext(np.array(negative_image))
#     # Find the detection with the highest probability above threshold
#     for (bbox, text, confidence) in all_reads:
#         if confidence > 0.5:
#             if best_result is None or confidence > best_result[2]:
                
#                 x1 = int(min([point[0] for point in bbox]))
#                 y1 = int(min([point[1] for point in bbox]))
#                 x2 = int(max([point[0] for point in bbox]))
#                 y2 = int(max([point[1] for point in bbox]))
#                 best_result = (bbox, text, confidence)
#                 result.append(best_result)

#     print(f"Detected license plates: {result}")
#     if best_result:
#         bbox, text, confidence = best_result
#         x1 = int(min([point[0] for point in bbox]))
#         y1 = int(min([point[1] for point in bbox]))
#         x2 = int(max([point[0] for point in bbox]))
#         y2 = int(max([point[1] for point in bbox]))

#         image_with_outline = img.copy()
#         imageDraw = ImageDraw.Draw(image_with_outline)
#         font = ImageFont.load_default()

#         imageDraw.rectangle([x1, y1, x2, y2], outline=255, width=3)

#         # Draw text above the plate
#         text_label = f"{text} ({confidence:.2f})"
#         text_x = x1
#         text_y = max(0, y1 - 25)
        
#         # Background for text
#         text_width = len(text_label) * 10
#         imageDraw.rectangle([text_x, text_y, text_x + text_width, text_y + 20], 
#                      fill=0, outline=255)
#         imageDraw.text((text_x + 2, text_y + 2), text_label, fill=(255, 255, 255), font=font)
#         image_with_outline.save("output.jpg")
        
        
#         return {"text": str(text), "confidence": float(confidence), "bbox": [x1, y1, x2, y2]}
#     return {"text": None, "confidence": None, "bbox": None}

# list_of_cars = [2,5,7]

# def detect_cars_in_image(image: Image.Image):
#     """Detect cars in image and return basic information"""
#     results = model(image)
    
#     car_count = 0
#     detected_cars = []
#     img_width, img_height = image.size

#     for r in results:
#         boxes = r.boxes
#         if boxes is None:
#             continue
            
#         for box in boxes:
#             cls_id = int(box.cls[0].item())
#             confidence = float(box.conf[0].item())
#             print('cls_id:', cls_id)

#             # Filter for objects in list_of_cars with sufficient confidence
#             if cls_id not in list_of_cars or confidence < CONF_THRESHOLD:
#                 continue

#             # Get bounding box coordinates
#             car_x1, car_y1, car_x2, car_y2 = map(int, box.xyxy[0].tolist())
            
#             # Ensure coordinates are within image bounds
#             car_x1, car_y1 = max(0, car_x1), max(0, car_y1)
#             car_x2, car_y2 = min(img_width, car_x2), min(img_height, car_y2)
            
#             # Skip if bounding box is invalid
#             if car_x2 <= car_x1 or car_y2 <= car_y1:
#                 continue
            
#             car_count += 1
#             car_id = f"car_{car_count}_{uuid.uuid4().hex[:8]}"
            
#             # Crop and save car image
#             car_crop = image.crop((car_x1, car_y1, car_x2, car_y2))
#             filename = f"{car_id}.jpg"
#             filepath = os.path.join(DETECTED_CAR_FOLDER, filename)
#             car_crop.save(filepath)
            
          
#             image_with_outline = image.copy()
#             imageDraw = ImageDraw.Draw(image_with_outline)
#             font = ImageFont.load_default()

#             imageDraw.rectangle([car_x1, car_y1, car_x2, car_y2], outline=255, width=3)
#             text_label = f"Confidence: ({confidence:.2f})"
#             text_x = car_x1
#             text_y = max(0, car_y1 - 25)
#             text_width = len(text_label) * 10
#             imageDraw.rectangle([text_x, text_y, text_x + text_width, text_y + 20], 
#                             fill=0, outline=255)
#             imageDraw.text((text_x + 2, text_y + 2), text_label, fill=(255, 255, 255), font=font)
#             outlined_img_path = os.path.join(DETECTED_CAR_FOLDER, "outlined_" + car_id + ".jpg")
#             image_with_outline.save(outlined_img_path)

#             # Basic car info
#             car_info = {
#                 "car_id": car_id,
#                 "confidence": round(confidence, 3),
#                 "bbox": {
#                     "x1": car_x1, "y1": car_y1, "x2": car_x2, "y2": car_y2,
#                     "width": car_x2 - car_x1,
#                     "height": car_y2 - car_y1
#                 },
#                 "image_file": filepath,
#                 "outlined_car_path": outlined_img_path
#             }
#             detected_cars.append(car_info)

#     return detected_cars

# def check_correct_text():
#     return None

# @app.post("/get_text_from_image")
# async def get_text_from_image(image: UploadFile = File(...)):
#     try:
#         contents = await image.read()
#         # # Convert to grayscale
#         img = Image.open(io.BytesIO(contents))
#         # # Apply bilateral filter
#         # gray_np = np.array(gray)
#         # filtered = cv2.bilateralFilter(gray_np, 11, 11, 17)
       
#         # # # Convert back to RGB for OCR (if needed)
#         # img = Image.fromarray(filtered).convert("RGB")
#         # img.save("processed_image.jpg")
#         # img = cv2.imread(contents)
#         # print('img:', img)
        
   
#         print('@@@@@@@@@@')
#         equ = cv2.equalizeHist(img)
#         print('equalized:', equ)


#         output = reader.readtext(equ)
        

#     except Exception:
#         raise HTTPException(400, "Invalid image")

#     all_text = reader.readtext(np.array(output))
#     print('all_text:', all_text)
#     return None


from ultralytics import YOLO

model = YOLO("yolo11l.pt")

train_results = model.train(
    data="data.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    # device='cpu'
)

metrics = model.val()   # evaluate
model.predict("samples/", conf=0.25, save=True)  # inference
model.export(format="onnx")  # export