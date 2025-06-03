import os
import cv2

# Recarregar o vídeo após reset
video_path = "video/6_65__Amazon Warehouse Order Picking Robots.mp4"
output_dir = "video/frames"

# Criar diretório de saída
os.makedirs(output_dir, exist_ok=True)

# Abrir vídeo com OpenCV
cap = cv2.VideoCapture(video_path)
frame_count = 0
frame_rate = 10  # extrair 10 frames por segundo
fps = cap.get(cv2.CAP_PROP_FPS)
step = int(fps // frame_rate) if fps > frame_rate else 1

# Extrair frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % step == 0:
        resized = cv2.resize(frame, (500, 400))  # ajusta à janela do menu
        filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(filename, resized)
    frame_count += 1

cap.release()

# Mostrar os primeiros frames extraídos
sorted(os.listdir(output_dir))[:5]
