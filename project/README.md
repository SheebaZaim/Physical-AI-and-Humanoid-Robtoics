# Educational Book Platform for Physical AI & Humanoid Robotics

A visual-first AI-assisted book platform for Physical AI & Humanoid Robotics, featuring:

- Interactive documentation with visual elements
- AI-powered RAG chatbot for answering questions
- Personalized learning experience
- Urdu translation capabilities
- Comprehensive coverage of ROS 2, Gazebo/Unity, NVIDIA Isaac, VLA, and Capstone projects

## Tech Stack

- **Frontend**: Docusaurus with custom theming
- **Backend**: FastAPI
- **Database**: Neon Serverless Postgres
- **Vector Database**: Qdrant Cloud
- **AI Services**: OpenAI Agents / ChatKit SDKs

## Getting Started

### Prerequisites

- Node.js 18+ for frontend
- Python 3.11+ for backend
- Git

### Installation

1. Clone the repository
2. Navigate to the frontend directory and install dependencies: `npm install`
3. Navigate to the backend directory and install dependencies: `pip install -r requirements.txt`
4. Start the development servers

### Frontend Development

```bash
cd frontend
npm start
```

### Backend Development

```bash
cd backend
uvicorn main:app --reload
```

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

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.