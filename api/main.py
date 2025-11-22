import io
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import uvicorn

# Import functions from existing scripts
from server.main import collect_all_reviews
from sentiment.main import analyze_sentiment_full
from quantitative.main import predict_themes_df

app = FastAPI(title="ReviewNet API", description="API for app review analysis", version="1.0.0")

@app.post("/scrape-reviews")
async def scrape_reviews(app_id: str = Form(...), lang: str = Form("en"), country: str = Form("us")):
    try:
        # Create a temporary file for output
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            output_file = tmp.name

        # Call the scraping function
        collect_all_reviews(app_id, output_file, lang, country)

        # Read the CSV and return as JSON
        df = pd.read_csv(output_file)
        os.unlink(output_file)  # Clean up

        return {"reviews": df['content'].tolist(), "count": len(df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-sentiment")
async def analyze_sentiment_endpoint(file: UploadFile = File(...), text_column: str = Form("content")):
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{text_column}' not found")

        # Use the full analysis function
        results = analyze_sentiment_full(df, text_column)

        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            output_file = tmp.name
            results.to_csv(output_file, index=False)

        # Return file response
        return FileResponse(output_file, media_type='text/csv', filename="sentiment_analysis.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-themes")
async def classify_themes_endpoint(file: UploadFile = File(...), text_column: str = Form("content")):
    try:
        # Read uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if text_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{text_column}' not found")

        # Use the theme classification function
        results = predict_themes_df(df.copy(), text_column)

        # Create temporary file for download
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            output_file = tmp.name
            results.to_csv(output_file, index=False)

        # Return file response
        return FileResponse(output_file, media_type='text/csv', filename="theme_classification.csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
