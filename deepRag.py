import os
from typing import List, Dict, Any
import numpy as np
from dataclasses import dataclass
from neo4j import GraphDatabase
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from sentence_transformers import SentenceTransformer
import json
import logging
from pathlib import Path
from datetime import datetime
import asyncio
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Represents a document with its content and metadata."""
    content: str
    filepath: str
    metadata: Dict[Any, Any]
    embedding: np.ndarray = None

class Neo4jKnowledgeGraph:
    """Manages document relationships and metadata in Neo4j."""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self._init_schema()
    
    def _init_schema(self):
        """Initialize Neo4j schema and constraints."""
        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.filepath IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (f:Folder) REQUIRE f.path IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patch) REQUIRE p.hash IS UNIQUE",
                "CREATE CONSTRAINT IF NOT EXISTS FOR (c:CodeChange) REQUIRE (c.file, c.line) IS UNIQUE"
            ]
            
            # Create indexes
            indexes = [
                "CREATE INDEX document_content IF NOT EXISTS FOR (d:Document) ON (d.content)",
                "CREATE INDEX patch_content IF NOT EXISTS FOR (p:Patch) ON (p.description)",
                "CREATE INDEX folder_path IF NOT EXISTS FOR (f:Folder) ON (f.path)"
            ]
            
            for constraint in constraints:
                session.run(constraint)
            for index in indexes:
                session.run(index)
    
    def add_document(self, doc: Document, folder_path: str = ""):
        """Add a document and maintain folder hierarchy."""
        with self.driver.session() as session:
            session.execute_write(self._create_document_with_folders, doc, folder_path)
    
    def _create_document_with_folders(self, tx, doc: Document, folder_path: str):
        """Create document node and connect it to folder hierarchy."""
        if folder_path:
            # Create folder hierarchy
            folders = folder_path.split('/')
            current_path = ""
            
            # Create root folder if it doesn't exist
            if folders[0]:
                root_query = """
                    MERGE (f:Folder {path: $path})
                    RETURN f
                """
                tx.run(root_query, path=folders[0])
                current_path = folders[0]
            
            # Create folder hierarchy
            for folder in folders[1:]:
                if folder:
                    parent_path = current_path
                    current_path = f"{current_path}/{folder}" if current_path else folder
                    folder_query = """
                        MATCH (parent:Folder {path: $parent_path})
                        MERGE (child:Folder {path: $current_path})
                        MERGE (parent)-[:CONTAINS]->(child)
                    """
                    tx.run(folder_query, parent_path=parent_path, current_path=current_path)
        
        # Create document and connect to its folder
        doc_query = """
            MERGE (d:Document {
                filepath: $filepath,
                content: $content,
                file_type: $file_type,
                last_modified: $last_modified
            })
            SET d += $metadata
        """
        
        if folder_path:
            doc_query += """
            WITH d
            MATCH (f:Folder {path: $folder_path})
            MERGE (f)-[:CONTAINS]->(d)
            """
        
        # Add document metadata
        metadata = {
            **doc.metadata,
            'file_type': Path(doc.filepath).suffix,
            'last_modified': doc.metadata.get('last_modified', ''),
            'size_bytes': doc.metadata.get('size_bytes', 0)
        }
        
        tx.run(doc_query,
            folder_path=folder_path,
            filepath=doc.filepath,
            content=doc.content,
            file_type=metadata['file_type'],
            last_modified=metadata['last_modified'],
            metadata=metadata
        )
    
    def add_patch(self, patch_info: dict):
        """Add Git patch information to the graph."""
        with self.driver.session() as session:
            session.execute_write(self._create_patch_node, patch_info)
    
    def _create_patch_node(self, tx, patch_info: dict):
        """Create patch node and related changes."""
        # Create patch node
        patch_query = """
            MERGE (p:Patch {hash: $hash})
            SET p.author = $author,
                p.date = $date,
                p.description = $description,
                p.branch = $branch
        """
        
        tx.run(patch_query,
            hash=patch_info['hash'],
            author=patch_info['author'],
            date=patch_info['date'],
            description=patch_info['description'],
            branch=patch_info.get('branch', 'main')
        )
        
        # Add file changes
        for change in patch_info['changes']:
            change_query = """
                MATCH (p:Patch {hash: $hash})
                MERGE (c:CodeChange {
                    file: $filepath,
                    line: $line_number
                })
                SET c.type = $change_type,
                    c.content = $content
                MERGE (p)-[:MODIFIES]->(c)
                WITH c
                OPTIONAL MATCH (d:Document {filepath: $filepath})
                FOREACH (doc IN CASE WHEN d IS NOT NULL THEN [d] ELSE [] END |
                    MERGE (c)-[:IN_FILE]->(doc)
                )
            """
            
            tx.run(change_query,
                hash=patch_info['hash'],
                filepath=change['file'],
                line_number=change['line'],
                change_type=change['type'],
                content=change['content']
            )

def parse_git_patch(patch_content: str) -> dict:
    """Parse Git patch content into structured data."""
    import re
    
    # Basic patch info parsing
    header_pattern = re.compile(r'From\s([a-f0-9]+).*?\nFrom:\s(.*?)\nDate:\s(.*?)\n', re.DOTALL)
    header_match = header_pattern.search(patch_content)
    
    if not header_match:
        return None
    
    patch_info = {
        'hash': header_match.group(1),
        'author': header_match.group(2).strip(),
        'date': header_match.group(3).strip(),
        'description': '',
        'changes': []
    }
    
    # Extract commit message
    msg_pattern = re.compile(r'Subject:\s\[PATCH\](.*?)\n---', re.DOTALL)
    msg_match = msg_pattern.search(patch_content)
    if msg_match:
        patch_info['description'] = msg_match.group(1).strip()
    
    # Parse file changes
    diff_pattern = re.compile(r'diff --git a/(.*?) b/(.*?)\n.*?@@\s-(\d+),?\d*\s\+(\d+),?\d*\s@@', re.DOTALL)
    for diff_match in diff_pattern.finditer(patch_content):
        file_path = diff_match.group(2)  # Use the 'b' path as current
        start_line = int(diff_match.group(4))  # Use the '+' line number
        
        # Extract changed content
        changes_pattern = re.compile(r'@@.*?@@(.*?)(?=diff --git|\Z)', re.DOTALL)
        changes_match = changes_pattern.search(patch_content, diff_match.end())
        if changes_match:
            lines = changes_match.group(1).splitlines()
            current_line = start_line
            
            for line in lines:
                if line.startswith('+'):
                    patch_info['changes'].append({
                        'file': file_path,
                        'line': current_line,
                        'type': 'addition',
                        'content': line[1:].strip()
                    })
                    current_line += 1
                elif line.startswith('-'):
                    patch_info['changes'].append({
                        'file': file_path,
                        'line': current_line,
                        'type': 'deletion',
                        'content': line[1:].strip()
                    })
                else:
                    current_line += 1
    
    return patch_info

class EnhancedRetriever:
    """Enhanced retrieval system combining FAISS and Neo4j."""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 neo4j_uri: str = "bolt://localhost:7687",
                 neo4j_user: str = "neo4j",
                 neo4j_password: str = "password"):
        
        self.embedding_model = SentenceTransformer(embedding_model)
        self.knowledge_graph = Neo4jKnowledgeGraph(
            neo4j_uri, neo4j_user, neo4j_password
        )
        self.index = None
        self.documents = []
        
        # Supported file types
        self.supported_files = {'.txt', '.md', '.py', '.patch'}
        
        # Initialize or load document tracking
        self.processed_files = self._load_processed_files()
    
    def _load_processed_files(self) -> set:
        """Load the set of already processed files."""
        try:
            with open('.processed_files.json', 'r') as f:
                return set(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return set()
    
    def _save_processed_files(self):
        """Save the set of processed files."""
        with open('.processed_files.json', 'w') as f:
            json.dump(list(self.processed_files), f)
    
    def add_documents(self, directory: str, force_reload: bool = False):
        """Load and process documents from a directory."""
        documents = []
        embeddings = []
        new_files_processed = False
        
        # Check if directory exists
        directory_path = Path(directory)
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        # Find all valid files
        valid_files = list(directory_path.rglob("*"))
        if not valid_files:
            raise ValueError(f"No files found in directory: {directory}")
            
        logger.info(f"Found {len(valid_files)} files in directory")
        
        for filepath in valid_files:
            if not filepath.is_file() or filepath.suffix not in self.supported_files:
                continue
            
            # Skip if file has been processed before and force_reload is False
            file_key = f"{filepath.absolute()}:{filepath.stat().st_mtime}"
            if not force_reload and file_key in self.processed_files:
                logger.info(f"Skipping already processed file: {filepath}")
                continue
                
            try:
                # Get file metadata
                stat = filepath.stat()
                metadata = {
                    'filename': filepath.name,
                    'file_type': filepath.suffix,
                    'size_bytes': stat.st_size,
                    'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                }
                
                # Process patch files differently
                if filepath.suffix == '.patch':
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        patch_content = f.read()
                        patch_info = parse_git_patch(patch_content)
                        if patch_info:
                            self.knowledge_graph.add_patch(patch_info)
                            metadata['patch_hash'] = patch_info['hash']
                
                # Read file content
                content = filepath.read_text(errors='ignore')
                if not content.strip():
                    logger.warning(f"Empty file: {filepath}")
                    continue
                
                # Create document chunks
                chunks = self._chunk_document(content)
                logger.info(f"Created {len(chunks)} chunks for {filepath.name}")
                
                # Get relative path for folder structure
                rel_path = str(filepath.relative_to(directory_path).parent)
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        content=chunk,
                        filepath=str(filepath),
                        metadata={
                            **metadata,
                            'chunk_index': i,
                            'total_chunks': len(chunks),
                            'chunk_size': len(chunk)
                        }
                    )
                    
                    # Generate embedding
                    embedding = self.embedding_model.encode(chunk)
                    doc.embedding = embedding
                    
                    documents.append(doc)
                    embeddings.append(embedding)
                    
                    # Add to Neo4j with folder structure
                    try:
                        self.knowledge_graph.add_document(doc, rel_path)
                    except Exception as e:
                        logger.error(f"Failed to add document to Neo4j: {e}")
                        
                    # Mark file as processed
                    self.processed_files.add(file_key)
                    new_files_processed = True
                        
            except Exception as e:
                logger.error(f"Error processing file {filepath}: {e}")
                continue
        
        if not documents and not self.documents:
            raise ValueError("No valid documents were processed")
        
        if documents:  # Only update index if we have new documents
            # Update FAISS index
            embeddings_array = np.array(embeddings).astype('float32')
            embedding_dim = embeddings_array.shape[1]
            
            if self.index is None:
                self.index = faiss.IndexFlatL2(embedding_dim)
                logger.info(f"Created new FAISS index with dimension {embedding_dim}")
            
            self.index.add(embeddings_array)
            self.documents.extend(documents)
            
            logger.info(f"Successfully processed {len(documents)} new document chunks")
            
            # Save processed files list
            if new_files_processed:
                self._save_processed_files()
    
    def _chunk_document(self, content: str, chunk_size: int = 512) -> List[str]:
        """Split document into smaller chunks."""
        words = content.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def retrieve(self, query: str, k: int = 3) -> List[Document]:
        """Retrieve relevant documents using hybrid search."""
        if not self.documents:
            raise ValueError("No documents have been added to the retriever")
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # FAISS similarity search
        distances, indices = self.index.search(
            np.array([query_embedding]).astype('float32'), k
        )
        
        # Get relevant documents
        relevant_docs = [self.documents[i] for i in indices[0]]
        
        # If this is a diagnostic question, also check for related patches
        if any(word in query.lower() for word in ['error', 'issue', 'bug', 'fix', 'problem', 'diagnostic']):
            with self.knowledge_graph.driver.session() as session:
                # Query Neo4j for related patches
                result = session.run("""
                    MATCH (d:Document)-[:IN_FILE]-(c:CodeChange)-[:MODIFIES]-(p:Patch)
                    WHERE d.content CONTAINS $query
                    RETURN p.description as patch_desc,
                           collect(c.content) as changes,
                           p.date as date
                    ORDER BY p.date DESC
                    LIMIT 3
                """, query=query)
                
                # Add patch information to relevant documents
                for record in result:
                    patch_doc = Document(
                        content=f"Related Fix:\nDate: {record['date']}\nDescription: {record['patch_desc']}\nChanges:\n" + "\n".join(record['changes']),
                        filepath="patch_info",
                        metadata={'type': 'patch_info'}
                    )
                    relevant_docs.append(patch_doc)
        
        return relevant_docs

class DeepSeekRAG:
    """Main RAG system using DeepSeek model."""
    
    def __init__(self, retriever: EnhancedRetriever):
        self.retriever = retriever
        self.prompt_template = """
Context information is below.
---------------------
{context}
---------------------
Given the context information and no other information, answer the following question:
{question}

Answer:
"""
    
    def generate_answer(self, question: str) -> str:
        """Generate an answer using the DeepSeek model."""
        try:
            # Retrieve relevant context
            relevant_docs = self.retriever.retrieve(question)
            context = "\n".join(doc.content for doc in relevant_docs)
            
            # Prepare the prompt
            prompt = self.prompt_template.format(
                context=context,
                question=question
            )

            # Validate that ollama is running
            try:
                import requests
                requests.get("http://localhost:11434/api/health", timeout=1)
            except requests.exceptions.RequestException:
                return "Error: Unable to connect to ollama. Please ensure the ollama server is running."
            
            # Call DeepSeek model using ollama
            response = ollama.generate(
                model="deepseek-r1:1.5b",
                prompt=prompt
            )
            
            # Extract the response text - ollama.generate returns a dict with 'response' key
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error generating the response: {str(e)}"

def setup_logging(log_dir="./logs"):
    """Set up logging to both console and file."""
    import time
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create filename using Unix timestamp
    timestamp = int(time.time())
    log_file = os.path.join(log_dir, f"{timestamp}.log")
    
    # Configure logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatters and add it to the handlers
    file_formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add the handlers to the logger
    logger.addHandler(file_handler)
    
    logger.info(f"Starting new conversation at {datetime.now()}")
    return log_file

async def interactive_qa(retriever: EnhancedRetriever):
    """Run an interactive Q&A session."""
    print("\nWelcome to the Interactive Q&A System!")
    print("Type 'exit', 'quit', or press Ctrl+C to end the session.")
    print("Type 'clear' to clear the screen.")
    print("Type 'help' for commands.\n")
    
    rag = DeepSeekRAG(retriever)
    
    while True:
        try:
            # Get user input
            question = input("\nEnter your question: ").strip()
            
            # Log the question
            logger.info(f"Q: {question}")
            
            # Check for exit commands
            if question.lower() in ['exit', 'quit']:
                print("Goodbye!")
                logger.info("Session ended by user")
                break
                
            # Check for clear screen command
            elif question.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
                
            # Check for help command
            elif question.lower() == 'help':
                print("\nAvailable commands:")
                print("- exit/quit: End the session")
                print("- clear: Clear the screen")
                print("- help: Show this help message")
                continue
                
            # Process empty input
            elif not question:
                print("Please enter a question.")
                continue
            
            # Process the question
            print("\nThinking...")
            answer = rag.generate_answer(question)
            
            # Display and log the answer
            print(f"\nA: {answer}")
            logger.info(f"A: {answer}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            logger.info("Session ended by KeyboardInterrupt")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            print(f"\nAn unexpected error occurred: {e}")

async def main():
    """Main function that sets up and runs the RAG system."""
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="RAG System with document processing")
    parser.add_argument("--docs", type=str, help="Path to documents directory")
    parser.add_argument("--force-reload", action="store_true", 
                       help="Force reload all documents even if previously processed")
    args = parser.parse_args()
    
    # Set up logging
    log_file = setup_logging()
    logger.info(f"Logging to: {log_file}")
    
    # Get the documents directory
    docs_dir = args.docs or os.getenv("DOCS_DIR", "./documents")
    
    try:
        # Initialize the retriever
        retriever = EnhancedRetriever(
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        # Load documents if directory is provided
        if docs_dir and Path(docs_dir).exists():
            logger.info(f"Loading documents from: {docs_dir}")
            retriever.add_documents(docs_dir, force_reload=args.force_reload)
        else:
            logger.info("No documents directory provided or directory doesn't exist. Starting with existing index.")
        
        # Start interactive session
        await interactive_qa(retriever)
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())