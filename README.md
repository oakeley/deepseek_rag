# DeepSeek RAG System

A Retrieval-Augmented Generation (RAG) system that combines document analysis, knowledge graphs, and the DeepSeek language model for intelligent question answering.

## Features

- Document processing and embedding using FAISS
- Knowledge graph storage using Neo4j
- Integration with DeepSeek via Ollama
- Conversation history tracking
- Intelligent context retrieval
- Support for multiple file types (.txt, .md, .py, .patch)

## Prerequisites

- Python 3.12
- Neo4j Database Server
- Ollama with DeepSeek model

## Installation

1. Install system dependencies:
```bash
# Install Neo4j (Ubuntu/Debian)
sudo apt install neo4j
docker-compose -f docker-compose-neo4j.yml up

# Install Ollama
curl https://ollama.ai/install.sh | sh
```

2. Install the DeepSeek model in Ollama:
```bash
ollama pull deepseek-r1:1.5b
```

3. Create and activate Conda environment:
```bash
conda create -n deepseek python=3.12
conda activate deepseek
```

4. Install PyTorch and FAISS using Conda:
```bash
# Nvidia GPU support (replace '12.1' with your CUDA version)
conda install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1
conda install -c conda-forge faiss-gpu

# Or for CPU-only version (verrrry slow)
# conda install -c pytorch -c nvidia pytorch torchvision torchaudio cpuonly
# conda install -c conda-forge faiss-cpu
```


5. Install other required packages via pip:
```bash
pip install sentence-transformers \
            transformers \
            neo4j \
            numpy \
            pandas \
            requests \
            python-dotenv \
            httpx \
            ollama
```


## Configuration

1. Always start Neo4j in a window:
```bash
# Start Neo4j service
docker-compose -f docker-compose-neo4j.yml up
```

2. Set environment variables (optional):
```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export DOCS_DIR="./documents"
```

## Usage

### Basic Usage

Run the script with default settings:
```bash
python deepRag.py
```

### Command Line Options

1. Specify documents directory:
```bash
python deepRag.py --docs /path/to/documents
```

2. Force reload all documents:
```bash
python deepRag.py --docs /path/to/documents --force-reload
```

### Interactive Commands

When running the system:
- Type your questions and press Enter
- Type 'exit' or 'quit' to end the session
- Type 'clear' to clear the screen
- Type 'help' for commands list

## Directory Structure

```
.
├── documents/          # Directory for source documents
├── logs/              # System logs
├── rag_system.py      # Main script
├── requirements.txt   # Python dependencies
└── .processed_files.json  # Tracks processed files
```

## Log Files

- Log files are stored in `./logs/` directory
- Each session creates a new log file with Unix timestamp
- Logs include all Q&A interactions and system events

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| NEO4J_URI | Neo4j connection URI | bolt://localhost:7687 |
| NEO4J_USER | Neo4j username | neo4j |
| NEO4J_PASSWORD | Neo4j password | password |
| DOCS_DIR | Documents directory | ./documents |

## Example Session

```bash
$ python deepRag.py --docs ./my_project
Welcome to the Interactive Q&A System!
Type 'exit', 'quit', or press Ctrl+C to end the session.
Type 'clear' to clear the screen.
Type 'help' for commands.

Enter your question: How does the bolus calculation work?

[System processes question using available documentation and history]
```

## Important Notes

1. The system maintains a knowledge base between sessions:
   - Previously processed documents are tracked
   - Q&A history is preserved
   - Document embeddings are stored

2. Neo4j must be running for the system to work.

3. Ollama must be running with the DeepSeek model installed.

4. First run may take longer due to initial document processing.

## Troubleshooting

1. If Neo4j connection fails:
   - Ensure Neo4j service is running
   - Verify credentials
   - Check connection URI

2. If Ollama fails:
   - Ensure Ollama service is running
   - Verify DeepSeek model is installed
   - Check Ollama logs

3. If document processing fails:
   - Check file permissions
   - Verify supported file types
   - Check available disk space

## Support

For issues and questions:
1. Check logs in ./logs directory
2. Ensure all prerequisites are installed
3. Verify environment variables
