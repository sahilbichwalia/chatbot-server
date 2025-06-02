from langchain.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from langchain.embeddings import HuggingFaceEmbeddings
from src.common.constant import BASE_MODEL,EMBEDDING_MODEL
from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
import os
load_dotenv()  # Load variables from .env into environment

api_key = os.getenv("api_key")
os.environ["GOOGLE_API_KEY"] = api_key

LLM_INSTANCE = ChatOllama(
           model=BASE_MODEL,  # Or your preferred model
           temperature=0.05,   # Low temperature for more deterministic ReAct
           top_k=10,
           top_p=0.7,          # Tighter sampling
           request_timeout=120,  # Generous timeout
           max_iterations=3,
           system_message="You are a helpful AI assistant that handles system monitoring and sustainability queries with expertise."  # System greeting/context
)


# LLM_INSTANCE= ChatGoogleGenerativeAI(
#         model=BASE_MODEL,
#         temperature=0.5,
#         convert_system_message_to_human=True ,
#         top_k=30,
#         top_p=0.85,
#         #  request_timeout=120
# )
Embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'batch_size': 16, 'normalize_embeddings': True}  # Reduced batch size, enabled normalization
        )