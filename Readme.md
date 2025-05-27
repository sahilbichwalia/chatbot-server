# HPE CHATBOT

An intelligent AI assistant built with LangChain that supports both local Ollama models and Google Generative AI, specifically designed for HPE system monitoring and PDF document processing with sustainability focus.

## Features

- **Dual LLM Support**: Switch between local Ollama (gemma2:9b) and Google Generative AI
- **PDF Processing**: Handles HPE metrics and energy efficiency documents
- **Vector Database**: ChromaDB integration for document embeddings and retrieval
- **HuggingFace Embeddings**: Efficient text processing with sentence transformers
- **System Monitoring**: Specialized for HPE system monitoring and sustainability queries
- **Document Chunking**: Smart document splitting for better context management

## Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed locally
- Google API key for Generative AI
- HPE system access (for monitoring features)

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Setup Ollama

#### Install Ollama:
- **macOS/Linux**: 
  ```bash
  curl -fsSL https://ollama.ai/install.sh | sh
  ```
- **Windows**: Download from [ollama.ai](https://ollama.ai/)

#### Pull the Gemma2:9b Model:
```bash
ollama pull gemma2:9b
```

#### Verify Installation:
```bash
ollama list
```

### 3. Environment Configuration

Create a `.env` file in the project root:

```env
# Google API Configuration
api_key=your_google_api_key_here

# Model Configuration
BASE_MODEL=gemma2:9b
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

### 4. Project Structure

```
HPE CHATBOT/
├── src/
│   └── common/
│       ├── common.py
│       └── constant.py
├── data/
│   ├── new_dump1.json
│   ├── logs/
│   └── pdf/
│       ├── hpe_metrics.pdf
│       ├── HPE_Server_Energy_Efficiency_Knowledge_Base(1).pdf
│       └── HPE_Server_Energy_Efficiency_Knowledge_Base.pdf
├── chroma_db/
├── main.py
├── requirements.txt
├── .env
└── README.md
```

## Configuration

### Model Constants

Update `src/common/constant.py` with your preferred models:

```python
# Base LLM Model
BASE_MODEL = "gemma2:9b"  # For Ollama
# BASE_MODEL = "gemini-pro"  # For Google Generative AI

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### Current Setup (from common.py)

The project currently uses:

1. **Google Generative AI** (active configuration):
   ```python
   LLM_INSTANCE = ChatGoogleGenerativeAI(
       model=BASE_MODEL,
       temperature=0.5,
       convert_system_message_to_human=True,
       top_k=30,
       top_p=0.85,
   )
   ```

2. **Ollama Configuration** (commented out):
   ```python
   # LLM_INSTANCE = ChatOllama(
   #     model=BASE_MODEL,  # gemma2:9b
   #     temperature=0.05,
   #     top_k=10,
   #     top_p=0.7,
   #     request_timeout=120,
   #     max_iterations=3,
   #     system_message="You are a helpful AI assistant..."
   # )
   ```

3. **HuggingFace Embeddings**:
   ```python
   Embeddings = HuggingFaceEmbeddings(
       model_name=EMBEDDING_MODEL,
       model_kwargs={'device': 'cpu'},
       encode_kwargs={'batch_size': 16, 'normalize_embeddings': True}
   )
   ```

## Usage

### Running the Application

```bash
python main.py
```

### PDF Document Processing

The system can process HPE-related PDF documents stored in the `pdf/` directory:
- `hpe_metrics.pdf` - HPE system metrics and monitoring data
- `HPE_Server_Energy_Efficiency_Knowledge_Base.pdf` - Energy efficiency documentation
- Additional PDF documents as needed

### Data Management

- **ChromaDB**: Vector database stored in `chroma_db/` directory
- **JSON Data**: Processed data dumps in `data/new_dump1.json`
- **Logs**: Application logs stored in `data/logs/`

### Switching Between Models

#### To Use Ollama (Local):
1. Ensure Ollama is running: `ollama serve`
2. In `src/common/common.py`, comment out the Google AI configuration
3. Uncomment the ChatOllama configuration
4. Restart the application

#### To Use Google Generative AI (Current):
1. Ensure your Google API key is set in `.env`
2. Keep the current ChatGoogleGenerativeAI configuration active
3. Run the application

### Query Examples

The chatbot specializes in:
- **HPE System Monitoring**: "What are the current CPU metrics?"
- **Energy Efficiency**: "How can I optimize server energy consumption?"
- **Sustainability**: "What are HPE's sustainability initiatives?"
- **Document Queries**: "Search for information about server efficiency in the knowledge base"

## Model Parameters Explained

### Ollama Configuration
- **temperature**: 0.05 (low for deterministic responses)
- **top_k**: 10 (consider top 10 tokens)
- **top_p**: 0.7 (nucleus sampling)
- **request_timeout**: 120 seconds
- **max_iterations**: 3 (for ReAct pattern)

### Google Generative AI Configuration
- **temperature**: 0.5 (balanced creativity/determinism)
- **top_k**: 30 (consider top 30 tokens)
- **top_p**: 0.85 (nucleus sampling)
- **convert_system_message_to_human**: True

### Embeddings Configuration
- **model**: sentence-transformers/all-MiniLM-L6-v2
- **device**: CPU (change to 'cuda' if GPU available)
- **batch_size**: 16 (optimized for performance)
- **normalize_embeddings**: True (recommended for similarity search)

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**:
   ```bash
   # Start Ollama service
   ollama serve
   
   # Check if gemma2:9b is available
   ollama list
   ```

2. **Google API Key Error**:
   - Verify your API key in `.env` file
   - Check API quota and billing status
   - Ensure the key has proper permissions

3. **ChromaDB Issues**:
   ```bash
   # Clear ChromaDB if corrupted
   rm -rf chroma_db/
   # Restart application to rebuild
   ```

4. **PDF Processing Errors**:
   - Ensure PDF files are not corrupted
   - Check file permissions in `pdf/` directory
   - Verify sufficient disk space for processing

5. **Memory Issues**:
   - Reduce embedding batch_size from 16 to 8
   - Use smaller models if available
   - Enable GPU if available: `model_kwargs={'device': 'cuda'}`

6. **LangChain Deprecation Warnings**:
   - The terminal shows ChromaDB deprecation warnings
   - Update to newer versions or use `langchain-chroma` package
   - Current functionality remains unaffected

### Performance Optimization

1. **For GPU Usage**:
   ```python
   # In common.py, update embeddings configuration
   model_kwargs={'device': 'cuda'}
   ```

2. **For Faster Processing**:
   - Increase batch_size if memory allows
   - Use quantized models for Ollama
   - Enable persistent ChromaDB storage

3. **Memory Management**:
   - Monitor RAM usage during PDF processing
   - Clear vector store periodically
   - Use document chunking for large PDFs

## File Structure Details

### Key Files

- **`main.py`**: Main execution file - start your application here
- **`src/common/common.py`**: LLM and embeddings configuration
- **`src/common/constant.py`**: Model constants and configuration variables
- **`.env`**: Environment variables (create this file)
- **`requirements.txt`**: Python dependencies

### Data Directories

- **`data/`**: Contains processed data and logs
  - `new_dump1.json`: Preprocessed data dump
  - `logs/`: Application and error logs
- **`pdf/`**: HPE documentation and metrics files
- **`chroma_db/`**: Vector database storage (auto-generated)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with both Ollama and Google AI configurations
5. Update documentation if needed
6. Submit a pull request

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review terminal output for deprecation warnings
3. Verify all dependencies are installed correctly
4. Check Ollama documentation: [ollama.ai/docs](https://ollama.ai/docs)
5. Review LangChain documentation: [python.langchain.com](https://python.langchain.com/)

## Acknowledgments

- **HPE**: For system monitoring and sustainability focus
- **LangChain**: Framework for LLM applications
- **Ollama**: Local model serving with gemma2:9b
- **Google AI**: Generative AI capabilities
- **HuggingFace**: Sentence transformers for embeddings
- **ChromaDB**: Vector database for document storage