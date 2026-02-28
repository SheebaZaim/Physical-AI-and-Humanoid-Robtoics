---
title: Physical AI & Humanoid Robotics Book
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Physical AI & Humanoid Robotics Book Platform

Backend API for the Physical AI & Humanoid Robotics educational book platform.

Built with FastAPI + Qdrant + Gemini embeddings + OpenRouter (RAG chatbot).

## API Endpoints

- `GET /health` — Health check
- `POST /api/v1/chat/public-ask` — RAG chatbot (no auth required)
- `POST /api/v1/translation/translate` — Urdu translation

## Frontend

Deployed at Vercel: [Physical AI Book](https://physical-ai-and-humanoid-robtoics.vercel.app)
