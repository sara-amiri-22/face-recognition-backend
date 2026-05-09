import os
import time
import cv2
import requests
import numpy as np
from dotenv import load_dotenv
from supabase import create_client
from insightface.app import FaceAnalysis

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

face_app = FaceAnalysis(name="buffalo_s", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))


def download_image(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Could not decode image")

    return image


def extract_embedding(image):
    faces = face_app.get(image)

    if len(faces) == 0:
        raise ValueError("No face detected")

    largest_face = max(
        faces,
        key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
    )

    embedding = largest_face.normed_embedding.astype(float).tolist()

    if len(embedding) != 512:
        raise ValueError(f"Expected 512-dimensional embedding, got {len(embedding)}")

    return embedding


def process_pending_faces():
    result = (
        supabase
        .table("registered_user_faces")
        .select("id, face_image_url")
        .eq("embedding_status", "pending")
        .limit(10)
        .execute()
    )

    rows = result.data

    if not rows:
        print("No pending face rows found.")
        return

    for row in rows:
        face_id = row["id"]
        image_url = row["face_image_url"]

        try:
            print(f"Processing row: {face_id}")

            supabase.table("registered_user_faces").update({
                "embedding_status": "processing"
            }).eq("id", face_id).execute()

            image = download_image(image_url)
            embedding = extract_embedding(image)

            supabase.table("registered_user_faces").update({
                "face_embedding": embedding,
                "embedding_status": "completed"
            }).eq("id", face_id).execute()

            print(f"Completed row: {face_id}")

        except Exception as error:
            supabase.table("registered_user_faces").update({
                "embedding_status": "failed"
            }).eq("id", face_id).execute()

            print(f"Failed row {face_id}: {error}")


if __name__ == "__main__":
    import time

    while True:
        try:
            process_pending_faces()
        except Exception as e:
            print("Worker crashed:", e)

        time.sleep(10)