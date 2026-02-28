# Contributing to Educational Book Platform

We welcome contributions to the Educational Book Platform for Physical AI & Humanoid Robotics! This document provides guidelines for contributing to the project.

## Code of Conduct

Please follow our Code of Conduct in all interactions.

## How Can I Contribute?

### Reporting Bugs

- Use the issue tracker to report bugs
- Describe the issue clearly with reproduction steps
- Include environment information (OS, browser, etc.)

### Suggesting Features

- Use the issue tracker to suggest new features
- Explain the feature and why it would be useful
- Consider the project's scope and goals

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Add your changes with clear commit messages
4. Ensure your code follows the project's style guidelines
5. Submit a pull request with a clear description

## Development Setup

### Prerequisites

- Node.js 18+ for frontend
- Python 3.11+ for backend
- Git

### Frontend Development

```bash
cd frontend
npm install
npm start
```

### Backend Development

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

## Style Guidelines

### Git Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters or less
- Reference issues and pull requests after a blank line

### Frontend

- Follow the existing code style
- Use TypeScript for type safety
- Write meaningful component names
- Ensure responsive design

### Backend

- Follow PEP 8 style guide
- Use type hints for all functions
- Write docstrings for complex functions
- Handle errors appropriately

## Project Structure

```
project/
├── frontend/                 # Docusaurus site
│   ├── src/
│   │   ├── components/       # Reusable components
│   │   ├── pages/            # Static pages
│   │   └── theme/            # Custom theme overrides
│   ├── static/               # Static assets
│   ├── docs/                 # Book content
│   └── docusaurus.config.js  # Configuration
├── backend/                  # FastAPI application
│   ├── app/
│   │   ├── api/              # API routes
│   │   ├── models/           # Database models
│   │   ├── schemas/          # Pydantic schemas
│   │   ├── services/         # Business logic
│   │   └── core/             # Core configurations
│   ├── alembic/              # Database migrations
│   └── main.py               # Application entry point
├── ai/                       # AI integration modules
│   ├── embedding_engine/     # Embedding processing
│   ├── rag_chat/            # RAG chatbot logic
│   └── translation/         # Translation services
└── scripts/                 # Utility scripts
```

## Questions?

Feel free to reach out through the issue tracker if you have questions about contributing.