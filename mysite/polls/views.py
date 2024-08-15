import cv2
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
import os
from tempfile import NamedTemporaryFile
from django.conf import settings
from django.shortcuts import render



def upload_video(request):
    return render(request, 'front_end.html')

# Load the YOLO model
model = YOLO(r"C:\Users\Aleena Ajith\OneDrive\Desktop\DJANGO\mysite\yolov8n.pt")

@csrf_exempt
def detect_objects_from_video(request):
    if request.method == 'POST':
        video_file = request.FILES['video']
        with NamedTemporaryFile(delete=False) as temp_video:
            for chunk in video_file.chunks():
                temp_video.write(chunk)
            temp_video_path = temp_video.name

        # Open the video file
        cap = cv2.VideoCapture(temp_video_path)
        video_output_path = os.path.join(settings.MEDIA_ROOT, 'processed_video.mp4')
        
        # Ensure the media directory exists
        if not os.path.exists(settings.MEDIA_ROOT):
            os.makedirs(settings.MEDIA_ROOT)

        out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'avc1'), 30, (int(cap.get(3)), int(cap.get(4))))



        if not out.isOpened():
            return JsonResponse({'error': 'VideoWriter failed to open.'}, status=500)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Perform object detection on the frame
            results = model.predict(frame)

            # Draw bounding boxes and labels on the frame if needed
            for box in results[0].boxes:
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                label = model.names[int(box.cls)]
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            out.write(frame)

        cap.release()
        out.release()

        # Return the URL of the processed video
        video_url = os.path.join(settings.MEDIA_URL, 'processed_video.mp4').replace('\\', '/')
        return JsonResponse({'video_url': video_url})
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)