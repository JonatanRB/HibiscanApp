import os
import base64
import tempfile
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from hibiscus_ga_counter import process_image

app = FastAPI(title="Hibiscus Fruit Counter API", version="1.0.0")

# CORS para permitir conexión con tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Puedes restringir a tu dominio si deseas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 🔹 Función auxiliar para guardar archivo temporalmente
def save_upload_to_temp(upload: UploadFile) -> str:
    suffix = "." + (upload.filename.split(".")[-1] if "." in upload.filename else "png")
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(upload.file.read())
    return path


# 🔹 Endpoint principal
@app.post("/count")
async def count(
    file: UploadFile = File(...),
    optimize: bool = Query(False),
    annotate: bool = Query(False),
):
    try:
        temp_path = save_upload_to_temp(file)
        result = process_image(temp_path, optimize=optimize, save_json=False)

        payload = {
            "count": int(result["count"]),
            "hsv_ranges": result["hsv_ranges"],
            "min_area": result["min_area"],
            "max_area": result["max_area"],
        }

        # Si se solicita la imagen anotada
        if annotate:
            output_path = result.get("output_image")

            # 🔹 Convertir a ruta absoluta
            if output_path and not os.path.isabs(output_path):
                output_path = os.path.abspath(output_path)

            # 🔹 Verificar existencia del archivo
            if output_path and os.path.exists(output_path):
                print(f"✅ Imagen anotada generada: {output_path}")
                # Leer imagen como binario y convertir a Base64
                with open(output_path, "rb") as f:
                    b = f.read()
                payload["annotated_image_base64"] = base64.b64encode(b).decode("ascii")
            else:
                print(f"⚠️ No se encontró la imagen anotada: {output_path}")

        return JSONResponse(payload)

    except Exception as e:
        print("❌ Error en /count:", e)
        return JSONResponse({"error": str(e)}, status_code=400)


