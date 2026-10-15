import uvicorn

if __name__ == "__main__":
    uvicorn.run("turkce_llm_dlp.api:app", host="127.0.0.1", port=8000, access_log=False)
