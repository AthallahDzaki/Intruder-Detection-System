import cv2
import numpy as np
from ultralytics import YOLO

### Load model YOLOv8 yang telah dilatih
model = YOLO('yolov8n.pt')

### Fungsi untuk menambahkan teks di frame
def add_text(frame, text, position=(10, 30), color=(0, 0, 255), font_scale=0.7, thickness=2):
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

### Fungsi menghitung jarak antara pusat dua bounding box
def calculate_distance(box1, box2):
    # Menghitung pusat bounding box
    center1 = ((box1[0] + box1[2]) // 2, (box1[1] + box1[3]) // 2)
    center2 = ((box2[0] + box2[2]) // 2, (box2[1] + box2[3]) // 2)
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    return distance

### Fungsi deteksi intruder
def detect_intruder(source='video', video_path=None, output_path='output_intruder.avi', display_output=True, threshold=100):
    if source == 'realtime':
        cap = cv2.VideoCapture(0)
        print("Menggunakan kamera realtime untuk deteksi intruder...")
    else:
        if not video_path:
            raise ValueError("Path video harus diberikan jika source adalah 'video'.")
        cap = cv2.VideoCapture(video_path)
        print(f"Mendeteksi intruder dari video: {video_path}")
    
    if not cap.isOpened():
        raise FileNotFoundError("Error: Tidak bisa membuka sumber video.")

    cv2.namedWindow('Deteksi Intruder', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Deteksi Intruder', 640, 480)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video atau gagal membaca frame.")
            break

        # Deteksi menggunakan YOLO
        results = model(frame)
        annotated_frame = results[0].plot()
        person_boxes = []
        object_boxes = []

        # Ekstrak bounding box dari hasil deteksi
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bounding box
                label = model.names[int(box.cls)]

                if label == 'person':
                    person_boxes.append((x1, y1, x2, y2))  # Simpan bounding box "person"
                else:
                    object_boxes.append((x1, y1, x2, y2, label))  # Simpan bounding box objek lain

        # Tambahkan teks jika ada intruder
        if person_boxes:
            add_text(annotated_frame, "Intruder Detected", position=(10, 30), color=(0, 0, 255))

            # Periksa jarak antara "person" dan objek lain
            for person_box in person_boxes:
                for object_box in object_boxes:
                    distance = calculate_distance(person_box, object_box[:4])
                    print(f"Distance {distance}")
                    if distance < threshold:
                        add_text(annotated_frame, f"Intruder Detected and Near {object_box[4]}",
                                 position=(10, 60), color=(0, 255, 255))

        # Simpan dan tampilkan frame
        out.write(annotated_frame)
        if display_output:
            cv2.imshow('Deteksi Intruder', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Deteksi intruder selesai. Hasil disimpan di {output_path}")

### Contoh penggunaan
detect_intruder(source='video', video_path='video_demo.mp4', display_output=True, threshold=350)

# Untuk realtime (kamera):
# detect_intruder(source='realtime', display_output=True, batch_size=1, threshold=350)
