from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from anpr import detect_and_ocr

app = FastAPI(
    title="ANPR API",
    description="Automatic Number Plate Recognition API",
    version="1.0.0"
)


class ImageRequest(BaseModel):
    image_base64: str


@app.post("/api/detect")
async def detect_plate(req: ImageRequest):
    """
    Accepts:
        {
            "image_base64": "data:image/jpeg;base64,...."
        }
    Returns:
        200: {"plate": "MH12AB1234"}
        422: {"error": "No valid plate detected"}
        400: {"error": "Invalid image input"}
    """
    try:
        result = detect_and_ocr(req.image_base64)

        if result == "Invalid plate":
            # Semantic failure — image ok, but no plate detected
            return JSONResponse(
                {"error": "No valid plate detected"},
                status_code=422
            )

        return JSONResponse({"plate": result}, status_code=200)

    except Exception as e:
        # Likely malformed image or server issue
        return JSONResponse({"error": str(e)}, status_code=400)


@app.get("/")
async def root():
    return {"message": "ANPR API is running"}


if __name__ == "__main__":
    import uvicorn
    # uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ssl_keyfile="certs/key.pem",
        ssl_certfile="certs/cert.pem"
    )
