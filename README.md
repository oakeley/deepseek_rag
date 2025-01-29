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

Enter your question: how do the uploaded documents work?
INFO:__main__:Q: how do the uploaded documents work?

Thinking...
Batches: 100%|████████████████████████████████████| 1/1 [00:00<00:00, 48.23it/s]
INFO:httpx:HTTP Request: POST http://127.0.0.1:11434/api/generate "HTTP/1.1 200 OK"

A: <think>
Okay, so I'm trying to figure out how these documents work. From what I see on the document, it looks like each one is a PDF file with some kind of metadata attached, probably for tracking access or something. But there's also a lot of text at the bottom, which might be part of a record that wasn't fully rendered.

First, I think about databases. These seem to have titles, authors, creators, journals, and dates. They might store these details in an organized way so that people can find them easily. Maybe it's a library or academic database where researchers look up books or papers.

Then there are the metadata fields. Titles like "Title of the Document" and authors like "Author Name." I wonder if this is similar to book databases where you search by title or author for specific works. Also, journal names could be important for tracking when a document was published in certain journals.

The content inside the documents varies, but they all seem to have some relevant information. For example, some might include an abstract, which is like a summary of the paper's main points. This would help readers quickly understand what the document is about without reading the whole thing.

Looking at the dates, there are entries from 2013 and more recent years. Maybe these documents are part of a collection that grows over time. I'm guessing they might be part of a research project or a series where each entry builds on the previous one.

The text below all these documents is interesting too. It has something like "Record ID: ..." which probably refers to some unique identifier for each document, maybe in databases like ResearchGate. This helps in tracking and managing different versions if necessary.

I'm also thinking about how they might be used. If it's a library system, these could be cataloged books or e-books. Researchers might use them to find specific papers or topics quickly. For someone looking for a book on quantum mechanics, this database would have the titles and authors easily accessible.

Wait, but why is there extra text below the documents? Is that part of the metadata or something else? Maybe it's just additional information they added later, not crucial for searching. I should check how that's handled in databases to avoid confusion when someone tries to find specific documents.

I also recall seeing a section with an abstract after each document. That seems useful because it gives readers a quick idea of what the paper is about without reading the whole text. This would be handy for people who don't want to browse through the entire content.

Thinking deeper, these could be part of academic publications where each volume has specific information and references. Maybe they are e-journals or online articles that link back to their original sources. The metadata helps in linking these to their proper literature databases.

But wait, what if there's a mistake in the metadata? Like an old entry missing its ID or an incorrect date? That could be a problem for someone using this database. I should note that but assume everything is correct unless proven otherwise.

In summary, these documents are likely part of a structured database used for organizing academic and research materials. They provide essential metadata such as titles, authors, dates, and content summaries, aiding in efficient searching and tracking within the system.
</think>

**How Documents Work**

The documents presented are organized within a structured database system designed to facilitate efficient organization and retrieval of academic and research materials. Here's an overview of their functionality:

1. **Metadata Organization**: The database is organized by metadata fields such as:
   - **Title**: Provides clear names for each document.
   - **Author/Creator**: Identifies the person(s) who created or wrote the content.
   - **Journal/Journal Name**: Points to the publication or journal where the work was conducted.
   - **Date Range**: Offers a timeframe for the inclusion of works, aiding in tracking progression over time.

2. **Content Overview**: Each document contains relevant information:
   - **Abstract**: Summarizes the main points, providing a quick overview without reading the entire text.
   - **Content Details**: Includes titles and descriptions that help readers understand the scope and focus of each work.

3. **Usage Scenarios**:
   - **Research and Learning**: Researchers can easily search for specific papers or topics using metadata fields.
   - **Libraries and Databases**: Databases like ResearchGate utilize these entries to manage and track different versions of documents efficiently.

4. **Handling Metadata**: The database likely stores unique identifiers (ID) for each document, useful for tracking and managing variations or updates.

5. **Structure and Purpose**: These databases are akin to academic catalogs, where books and e-books are organized by titles and authors for quick accessibility. They serve as a central repository for academic publications, aiding researchers in finding relevant materials efficiently.

In essence, these documents are essential tools for managing and searching academic literature, providing both detailed metadata and accessible content summaries to enhance research and learning processes.
INFO:__main__:A: <think>
Okay, so I'm trying to figure out how these documents work. From what I see on the document, it looks like each one is a PDF file with some kind of metadata attached, probably for tracking access or something. But there's also a lot of text at the bottom, which might be part of a record that wasn't fully rendered.

First, I think about databases. These seem to have titles, authors, creators, journals, and dates. They might store these details in an organized way so that people can find them easily. Maybe it's a library or academic database where researchers look up books or papers.

Then there are the metadata fields. Titles like "Title of the Document" and authors like "Author Name." I wonder if this is similar to book databases where you search by title or author for specific works. Also, journal names could be important for tracking when a document was published in certain journals.

The content inside the documents varies, but they all seem to have some relevant information. For example, some might include an abstract, which is like a summary of the paper's main points. This would help readers quickly understand what the document is about without reading the whole thing.

Looking at the dates, there are entries from 2013 and more recent years. Maybe these documents are part of a collection that grows over time. I'm guessing they might be part of a research project or a series where each entry builds on the previous one.

The text below all these documents is interesting too. It has something like "Record ID: ..." which probably refers to some unique identifier for each document, maybe in databases like ResearchGate. This helps in tracking and managing different versions if necessary.

I'm also thinking about how they might be used. If it's a library system, these could be cataloged books or e-books. Researchers might use them to find specific papers or topics quickly. For someone looking for a book on quantum mechanics, this database would have the titles and authors easily accessible.

Wait, but why is there extra text below the documents? Is that part of the metadata or something else? Maybe it's just additional information they added later, not crucial for searching. I should check how that's handled in databases to avoid confusion when someone tries to find specific documents.

I also recall seeing a section with an abstract after each document. That seems useful because it gives readers a quick idea of what the paper is about without reading the whole text. This would be handy for people who don't want to browse through the entire content.

Thinking deeper, these could be part of academic publications where each volume has specific information and references. Maybe they are e-journals or online articles that link back to their original sources. The metadata helps in linking these to their proper literature databases.

But wait, what if there's a mistake in the metadata? Like an old entry missing its ID or an incorrect date? That could be a problem for someone using this database. I should note that but assume everything is correct unless proven otherwise.

In summary, these documents are likely part of a structured database used for organizing academic and research materials. They provide essential metadata such as titles, authors, dates, and content summaries, aiding in efficient searching and tracking within the system.
</think>

**How Documents Work**

The documents presented are organized within a structured database system designed to facilitate efficient organization and retrieval of academic and research materials. Here's an overview of their functionality:

1. **Metadata Organization**: The database is organized by metadata fields such as:
   - **Title**: Provides clear names for each document.
   - **Author/Creator**: Identifies the person(s) who created or wrote the content.
   - **Journal/Journal Name**: Points to the publication or journal where the work was conducted.
   - **Date Range**: Offers a timeframe for the inclusion of works, aiding in tracking progression over time.

2. **Content Overview**: Each document contains relevant information:
   - **Abstract**: Summarizes the main points, providing a quick overview without reading the entire text.
   - **Content Details**: Includes titles and descriptions that help readers understand the scope and focus of each work.

3. **Usage Scenarios**:
   - **Research and Learning**: Researchers can easily search for specific papers or topics using metadata fields.
   - **Libraries and Databases**: Databases like ResearchGate utilize these entries to manage and track different versions of documents efficiently.

4. **Handling Metadata**: The database likely stores unique identifiers (ID) for each document, useful for tracking and managing variations or updates.

5. **Structure and Purpose**: These databases are akin to academic catalogs, where books and e-books are organized by titles and authors for quick accessibility. They serve as a central repository for academic publications, aiding researchers in finding relevant materials efficiently.

In essence, these documents are essential tools for managing and searching academic literature, providing both detailed metadata and accessible content summaries to enhance research and learning processes.

Enter your question: 

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
