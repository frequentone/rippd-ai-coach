"""
Local-only server for the persona simulation eval harness.

Mirrors the /chat logic in main.py exactly, but skips Supabase JWT
verification — synthetic personas (user_0000, user_0001, ...) don't have
real Supabase auth accounts, so they can't produce a real JWT for
verify_user(). This is for local testing only; never deploy this file.

Run with: python eval_server.py  (from the ai/ directory, port 8001)
"""

import asyncio
from typing import Optional

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from context_builder import get_or_build_context, build_prompt_context, detect_topics
from rag_pipeline import chat

app = FastAPI(title="RIPPD Eval Server (no auth — local testing only)")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": exc.errors(), "body": exc.body})


@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


class ChatRequest(BaseModel):
    user_id: str
    message: str = Field(..., min_length=1, max_length=1000)
    creator_id: Optional[str] = None
    is_premium: bool = False


@app.get("/")
async def root():
    return {"status": "ok", "note": "eval-only server, no auth"}


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    slices = await asyncio.to_thread(get_or_build_context, req.user_id)
    topics = detect_topics(req.message)  # mirror /chat exactly, or evals lie
    prompt_context = build_prompt_context(slices, topics)
    return await asyncio.to_thread(
        chat,
        user_id=req.user_id,
        user_message=req.message,
        creator_id=req.creator_id,
        is_premium=req.is_premium,
        prompt_context=prompt_context,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
