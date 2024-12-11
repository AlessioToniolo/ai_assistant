from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app import ask_rag
import logging

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    messages: list


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"Received chat request with {len(request.messages)} messages")
        response = ask_rag(request.messages)
        logger.info("Successfully processed chat request")

        return JSONResponse(
            content={"response": response.content[0].text},
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
            },
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
