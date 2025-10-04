 

## About This Repository
This project establishes the skeleton for a Retrieval-Augmented Generation (RAG) pipeline using the LangChain framework to leverage a large PDF. It utilizes LangChain's document loaders and splitters for efficient data preparation and chunking. The system integrates an embedding model with a vector store to facilitate semantic search and retrieval. 
 

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/notAvailable73/RAG-Skeleton.git
cd RAG-Skeleton
```

2. **Create and activate a virtual environment**

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Setting up API Keys

Create a `.env` file in the root directory of the project and add your API key(s). There is a file named `.env.example` you can follow.

```
GROQ_API_KEY=your_groq_api_key 
```
 
### Test the environment

You'll find a file named `llm_test.ipynb`. Run the cells to see if everything is working or not. You'll find minor issues  if exists. (e.g. Credential errors)
 