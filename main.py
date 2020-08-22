from fastapi import FastAPI

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/masks/{img_url}")
def get_predicted_image(img_url: str):
    return f"Hello my {img_url}"

