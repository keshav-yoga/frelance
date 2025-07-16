# freelance_automation_final.py
"""
AI Freelance Automation System
A complete automation solution for freelance work management
"""

import os
import sys
import asyncio
import json
import logging
import hashlib
import secrets
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import pickle
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

# External imports with error handling
try:
    import aiohttp
    import aiofiles
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import yaml
    from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Text
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker, scoped_session
    from sqlalchemy.pool import QueuePool
    import redis
    from celery import Celery
    import gradio as gr
    from dotenv import load_dotenv
    import numpy as np
    from tenacity import retry, stop_after_attempt, wait_exponential
    
    # AI/ML imports
    import torch
    from transformers import (
        AutoTokenizer, 
        AutoModelForCausalLM,
        pipeline,
        BitsAndBytesConfig
    )
    from sentence_transformers import SentenceTransformer
    from langchain.llms import HuggingFacePipeline, Ollama
    from langchain.chains import LLMChain, ConversationChain
    from langchain.prompts import PromptTemplate
    from langchain.memory import ConversationBufferMemory
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.document_loaders import TextLoader, PDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Web automation
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    import undetected_chromedriver as uc
    from playwright.async_api import async_playwright
    
    # Additional utilities
    import schedule
    import pandas as pd
    from jinja2 import Template
    import markdown
    import pdfkit
    from PIL import Image
    import cv2
    import speech_recognition as sr
    from gtts import gTTS
    import edge_tts
    
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install all requirements: pip install -r requirements.txt")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure structured logging
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    grey = "\x1b[38;21m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
    }
    
    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Setup logging
def setup_logging(log_level=logging.INFO):
    """Configure application logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / f"freelance_automation_{datetime.now().strftime('%Y%m%d')}.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColoredFormatter())
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        handlers=[file_handler, console_handler]
    )
    
    # Suppress noisy libraries
    logging.getLogger("selenium").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

logger = setup_logging()

# Database Models
Base = declarative_base()

class Job(Base):
    __tablename__ = 'jobs'
    
    id = Column(String, primary_key=True)
    platform = Column(String, nullable=False)
    title = Column(String, nullable=False)
    description = Column(Text)
    budget_min = Column(Float)
    budget_max = Column(Float)
    deadline = Column(DateTime)
    skills = Column(Text)  # JSON string
    client_rating = Column(Float)
    status = Column(String, default='new')
    proposal_sent = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Project(Base):
    __tablename__ = 'projects'
    
    id = Column(String, primary_key=True)
    job_id = Column(String)
    platform = Column(String)
    status = Column(String, default='in_progress')
    deliverables = Column(Text)  # JSON string
    feedback = Column(Text)
    rating = Column(Float)
    earnings = Column(Float)
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

# Configuration Management
class Config:
    """Centralized configuration management"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            self._create_default_config()
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config['database']['url'] = os.getenv('DATABASE_URL', config['database']['url'])
        config['redis']['url'] = os.getenv('REDIS_URL', config['redis']['url'])
        
        return config
    
    def _create_default_config(self):
        """Create default configuration file"""
        default_config = {
            'app': {
                'name': 'AI Freelance Automation',
                'version': '1.0.0',
                'debug': False
            },
            'database': {
                'url': 'sqlite:///freelance_automation.db',
                'pool_size': 10,
                'max_overflow': 20
            },
            'redis': {
                'url': 'redis://localhost:6379/0',
                'ttl': 3600
            },
            'platforms': {
                'upwork': {
                    'enabled': True,
                    'api_url': 'https://www.upwork.com/api',
                    'rate_limit': 30
                },
                'fiverr': {
                    'enabled': True,
                    'api_url': 'https://api.fiverr.com',
                    'rate_limit': 60
                },
                'freelancer': {
                    'enabled': True,
                    'api_url': 'https://www.freelancer.com/api',
                    'rate_limit': 45
                }
            },
            'ai': {
                'models': {
                    'content': 'TheBloke/Mistral-7B-Instruct-v0.2-GGUF',
                    'code': 'TheBloke/CodeLlama-7B-Instruct-GGUF',
                    'embedding': 'sentence-transformers/all-MiniLM-L6-v2'
                },
                'ollama': {
                    'host': 'http://localhost:11434',
                    'models': ['mistral', 'codellama', 'llama2']
                },
                'max_tokens': 2048,
                'temperature': 0.7
            },
            'automation': {
                'scan_interval': 300,  # 5 minutes
                'max_concurrent_jobs': 5,
                'auto_bid': True,
                'min_match_score': 0.7
            },
            'security': {
                'encryption_key': Fernet.generate_key().decode(),
                'session_timeout': 3600,
                'max_login_attempts': 3
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
    
    def _validate_config(self):
        """Validate configuration values"""
        required_keys = [
            'app.name',
            'database.url',
            'ai.models.content',
            'security.encryption_key'
        ]
        
        for key in required_keys:
            if not self._get_nested_value(self.config, key):
                raise ValueError(f"Missing required configuration: {key}")
    
    def _get_nested_value(self, d: Dict, key: str) -> Any:
        """Get value from nested dictionary using dot notation"""
        keys = key.split('.')
        value = d
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._get_nested_value(self.config, key) or default

# Security Manager
class SecurityManager:
    """Handle encryption, authentication, and security"""
    
    def __init__(self, config: Config):
        self.config = config
        self.cipher_suite = Fernet(config.get('security.encryption_key').encode())
        self._setup_password_hasher()
    
    def _setup_password_hasher(self):
        """Setup password hashing with PBKDF2"""
        self.kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'stable_salt',  # In production, use random salt per password
            iterations=100000,
        )
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """Encrypt sensitive data"""
        if isinstance(data, str):
            data = data.encode()
        return self.cipher_suite.encrypt(data).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
    
    def hash_password(self, password: str) -> str:
        """Hash password for storage"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return self.hash_password(password) == hashed
    
    def generate_api_key(self) -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)

# Cache Manager
class CacheManager:
    """Redis-based caching system"""
    
    def __init__(self, config: Config):
        self.config = config
        self.redis_client = redis.from_url(
            config.get('redis.url'),
            decode_responses=True
        )
        self.default_ttl = config.get('redis.ttl', 3600)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis_client.get(key)
            if value:
                return json.loads(value)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        try:
            self.redis_client.setex(
                key,
                ttl or self.default_ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete value from cache"""
        self.redis_client.delete(key)
    
    def clear(self):
        """Clear all cache"""
        self.redis_client.flushdb()

# Rate Limiter
class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: int, per: int = 60):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = datetime.now()
        self.lock = threading.Lock()
    
    def __call__(self, func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not self.allow():
                wait_time = self.time_until_reset()
                logger.warning(f"Rate limit exceeded. Waiting {wait_time}s")
                await asyncio.sleep(wait_time)
            
            return await func(*args, **kwargs)
        return wrapper
    
    def allow(self) -> bool:
        """Check if action is allowed"""
        with self.lock:
            current = datetime.now()
            time_passed = (current - self.last_check).total_seconds()
            self.last_check = current
            
            self.allowance += time_passed * (self.rate / self.per)
            if self.allowance > self.rate:
                self.allowance = self.rate
            
            if self.allowance < 1.0:
                return False
            
            self.allowance -= 1.0
            return True
    
    def time_until_reset(self) -> float:
        """Time until next token is available"""
        return (1.0 - self.allowance) * (self.per / self.rate)

# AI Model Manager
class AIModelManager:
    """Manage AI models with optimizations"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize AI models with optimization"""
        try:
            # Use Ollama for local LLM inference
            self.models['content'] = Ollama(
                base_url=self.config.get('ai.ollama.host'),
                model="mistral",
                temperature=self.config.get('ai.temperature', 0.7)
            )
            
            self.models['code'] = Ollama(
                base_url=self.config.get('ai.ollama.host'),
                model="codellama",
                temperature=0.3
            )
            
            # Embedding model for semantic search
            self.models['embeddings'] = SentenceTransformer(
                self.config.get('ai.models.embedding'),
                device=self.device
            )
            
            # Initialize conversation memory
            self.memory = ConversationBufferMemory()
            
            logger.info("AI models initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            # Fallback to CPU-friendly models
            self._initialize_fallback_models()
    
    def _initialize_fallback_models(self):
        """Initialize smaller fallback models"""
        try:
            # Use smaller models that work on CPU
            self.models['content'] = pipeline(
                "text-generation",
                model="gpt2",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.models['embeddings'] = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device='cpu'
            )
            
            logger.info("Fallback models loaded")
        except Exception as e:
            logger.error(f"Failed to load fallback models: {e}")
    
    @lru_cache(maxsize=100)
    def generate_text(self, prompt: str, model_type: str = 'content', **kwargs) -> str:
        """Generate text with caching"""
        try:
            model = self.models.get(model_type)
            if not model:
                raise ValueError(f"Model {model_type} not found")
            
            if isinstance(model, Ollama):
                response = model.invoke(prompt)
                return response
            else:
                # Pipeline model
                response = model(prompt, **kwargs)
                return response[0]['generated_text']
                
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            return f"Error generating text: {str(e)}"
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate text embeddings"""
        return self.models['embeddings'].encode(text)
    
    def semantic_search(self, query: str, documents: List[str], top_k: int = 5) -> List[int]:
        """Perform semantic search"""
        query_embedding = self.embed_text(query)
        doc_embeddings = [self.embed_text(doc) for doc in documents]
        
        # Calculate cosine similarities
        similarities = []
        for doc_emb in doc_embeddings:
            sim = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append(sim)
        
        # Return top-k indices
        return np.argsort(similarities)[-top_k:][::-1].tolist()

# Platform API Clients
class PlatformClient:
    """Base class for platform API clients"""
    
    def __init__(self, platform: str, config: Config, security: SecurityManager):
        self.platform = platform
        self.config = config
        self.security = security
        self.session = None
        self.rate_limiter = RateLimiter(
            rate=config.get(f'platforms.{platform}.rate_limit', 60),
            per=60
        )
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _request(self, method: str, url: str, **kwargs) -> Dict:
        """Make HTTP request with retry logic"""
        async with self.session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()
    
    async def authenticate(self, credentials: Dict) -> bool:
        """Authenticate with platform"""
        raise NotImplementedError
    
    async def get_jobs(self, filters: Dict) -> List[Job]:
        """Get available jobs"""
        raise NotImplementedError
    
    async def submit_proposal(self, job_id: str, proposal: str) -> bool:
        """Submit proposal for job"""
        raise NotImplementedError
    
    async def get_messages(self) -> List[Dict]:
        """Get client messages"""
        raise NotImplementedError

class UpworkClient(PlatformClient):
    """Upwork API client implementation"""
    
    def __init__(self, config: Config, security: SecurityManager):
        super().__init__('upwork', config, security)
        self.api_url = config.get('platforms.upwork.api_url')
        self.oauth_token = None
    
    async def authenticate(self, credentials: Dict) -> bool:
        """OAuth2 authentication for Upwork"""
        try:
            # Implement OAuth2 flow
            # This is a placeholder - actual implementation requires OAuth2 library
            client_id = self.security.decrypt(credentials['client_id'])
            client_secret = self.security.decrypt(credentials['client_secret'])
            
            # Exchange credentials for token
            response = await self._request(
                'POST',
                f"{self.api_url}/oauth2/token",
                data={
                    'grant_type': 'client_credentials',
                    'client_id': client_id,
                    'client_secret': client_secret
                }
            )
            
            self.oauth_token = response.get('access_token')
            return bool(self.oauth_token)
            
        except Exception as e:
            logger.error(f"Upwork authentication failed: {e}")
            return False
    
    @RateLimiter(rate=30)
    async def get_jobs(self, filters: Dict) -> List[Job]:
        """Get jobs from Upwork"""
        try:
            headers = {'Authorization': f'Bearer {self.oauth_token}'}
            
            params = {
                'q': ' '.join(filters.get('skills', [])),
                'budget': f"[{filters.get('min_budget', 0)} TO {filters.get('max_budget', 10000)}]",
                'duration': filters.get('duration', 'week'),
                'job_type': filters.get('job_type', 'fixed'),
                'page': 0,
                'per_page': 50
            }
            
            response = await self._request(
                'GET',
                f"{self.api_url}/profiles/v2/search/jobs",
                headers=headers,
                params=params
            )
            
            jobs = []
            for job_data in response.get('jobs', []):
                job = Job(
                    id=job_data['id'],
                    platform='upwork',
                    title=job_data['title'],
                    description=job_data['description'],
                    budget_min=job_data.get('budget', {}).get('minimum'),
                    budget_max=job_data.get('budget', {}).get('maximum'),
                    deadline=datetime.fromisoformat(job_data.get('deadline', '')),
                    skills=json.dumps(job_data.get('skills', [])),
                    client_rating=job_data.get('client', {}).get('feedback', 0)
                )
                jobs.append(job)
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to get Upwork jobs: {e}")
            return []

# Web Automation Manager
class WebAutomationManager:
    """Handle web scraping and automation tasks"""
    
    def __init__(self, config: Config):
        self.config = config
        self.drivers = {}
        self._setup_drivers()
    
    def _setup_drivers(self):
        """Setup web drivers with anti-detection"""
        try:
            # Chrome options for stealth
            chrome_options = uc.ChromeOptions()
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Add user agent
            chrome_options.add_argument(
                'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            )
            
            # Create driver
            self.drivers['chrome'] = uc.Chrome(options=chrome_options)
            
        except Exception as e:
            logger.error(f"Failed to setup web driver: {e}")
    
    async def scrape_job_details(self, url: str) -> Dict:
        """Scrape additional job details from URL"""
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set user agent
                await page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                await page.goto(url, wait_until='networkidle')
                
                # Extract job details
                details = await page.evaluate('''() => {
                    const title = document.querySelector('h1')?.innerText || '';
                    const description = document.querySelector('.job-description')?.innerText || '';
                    const budget = document.querySelector('.budget')?.innerText || '';
                    const skills = Array.from(document.querySelectorAll('.skill-tag')).map(el => el.innerText);
                    
                    return { title, description, budget, skills };
                }''')
                
                await browser.close()
                return details
                
        except Exception as e:
            logger.error(f"Failed to scrape job details: {e}")
            return {}
    
    def cleanup(self):
        """Cleanup web drivers"""
        for driver in self.drivers.values():
            try:
                driver.quit()
            except:
                pass

# Work Generation Engine
class WorkGenerationEngine:
    """Generate high-quality work outputs"""
    
    def __init__(self, ai_manager: AIModelManager, config: Config):
        self.ai_manager = ai_manager
        self.config = config
        self.templates = self._load_templates()
        self.quality_checker = QualityChecker()
    
    def _load_templates(self) -> Dict[str, PromptTemplate]:
        """Load prompt templates"""
        templates_dir = Path("templates")
        templates_dir.mkdir(exist_ok=True)
        
        return {
            'proposal': PromptTemplate(
                input_variables=["job_title", "job_description", "skills", "portfolio"],
                template="""Write a professional and compelling freelance proposal for this job:

Job Title: {job_title}
Job Description: {job_description}
Required Skills: {skills}
My Portfolio: {portfolio}

The proposal should:
1. Start with a personalized greeting
2. Show clear understanding of the project requirements
3. Highlight relevant experience and skills
4. Propose a clear approach and timeline
5. Include competitive but fair pricing
6. End with a call to action

Make it sound natural, confident, and professional."""
            ),
            
            'content': PromptTemplate(
                input_variables=["topic", "style", "length", "keywords", "target_audience"],
                template="""Write {length} words of high-quality content about {topic}.

Style: {style}
Target Audience: {target_audience}
Keywords to include: {keywords}

Requirements:
- Engaging and informative
- Well-structured with clear sections
- SEO-optimized
- Original and plagiarism-free
- Conversational yet professional tone"""
            ),
            
            'code': PromptTemplate(
                input_variables=["language", "requirements", "constraints"],
                template="""Write {language} code that meets these requirements:

Requirements: {requirements}
Constraints: {constraints}

The code should be:
- Clean and well-commented
- Following best practices
- Efficient and optimized
- Fully functional
- Include error handling"""
            )
        }
    
    async def generate_proposal(self, job: Job, profile: Dict) -> str:
        """Generate customized proposal"""
        try:
            # Analyze job requirements
            job_skills = json.loads(job.skills) if job.skills else []
            matching_skills = list(set(job_skills) & set(profile.get('skills', [])))
            
            # Generate proposal
            proposal = self.ai_manager.generate_text(
                self.templates['proposal'].format(
                    job_title=job.title,
                    job_description=job.description,
                    skills=", ".join(matching_skills),
                    portfolio=profile.get('portfolio_summary', '')
                ),
                model_type='content'
            )
            
            # Enhance and humanize
            proposal = await self.humanize_text(proposal)
            
            # Quality check
            if self.quality_checker.check_proposal(proposal):
                return proposal
            else:
                # Regenerate if quality is low
                return await self.generate_proposal(job, profile)
                
        except Exception as e:
            logger.error(f"Failed to generate proposal: {e}")
            return ""
    
    async def generate_content(self, content_type: str, requirements: Dict) -> str:
        """Generate various types of content"""
        try:
            if content_type == "blog_post":
                return await self._generate_blog_post(requirements)
            elif content_type == "ebook":
                return await self._generate_ebook(requirements)
            elif content_type == "code":
                return await self._generate_code(requirements)
            elif content_type == "website":
                return await self._generate_website(requirements)
            else:
                raise ValueError(f"Unknown content type: {content_type}")
                
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            return ""
    
    async def _generate_blog_post(self, requirements: Dict) -> str:
        """Generate SEO-optimized blog post"""
        content = self.ai_manager.generate_text(
            self.templates['content'].format(
                topic=requirements.get('topic'),
                style=requirements.get('style', 'informative'),
                length=requirements.get('word_count', 1000),
                keywords=", ".join(requirements.get('keywords', [])),
                target_audience=requirements.get('audience', 'general')
            ),
            model_type='content',
            max_length=requirements.get('word_count', 1000) + 200
        )
        
        # Add formatting
        formatted = self._format_blog_post(content)
        
        # Generate meta description
        meta_desc = self.ai_manager.generate_text(
            f"Write a 150-character meta description for this blog post: {content[:200]}...",
            model_type='content',
            max_length=200
        )
        
        return {
            'content': formatted,
            'meta_description': meta_desc,
            'word_count': len(content.split())
        }
    
    async def _generate_code(self, requirements: Dict) -> str:
        """Generate code with documentation"""
        code = self.ai_manager.generate_text(
            self.templates['code'].format(
                language=requirements.get('language', 'python'),
                requirements=requirements.get('description'),
                constraints=requirements.get('constraints', 'None')
            ),
            model_type='code'
        )
        
        # Add proper formatting and syntax highlighting
        return self._format_code(code, requirements.get('language'))
    
    async def humanize_text(self, text: str) -> str:
        """Make AI-generated text sound more human"""
        humanization_prompt = f"""Rewrite this text to sound more natural and human-written:

{text}

Guidelines:
- Vary sentence structure and length
- Add personal touches and conversational elements
- Use contractions where appropriate
- Include transitional phrases
- Avoid repetitive patterns
- Maintain the original meaning and key points"""
        
        humanized = self.ai_manager.generate_text(
            humanization_prompt,
            model_type='content',
            temperature=0.8
        )
        
        return humanized
    
    def _format_blog_post(self, content: str) -> str:
        """Format blog post with proper structure"""
        # Split into paragraphs
        paragraphs = content.split('\n\n')
        
        # Add headings
        formatted = []
        for i, para in enumerate(paragraphs):
            if i == 0:
                formatted.append(f"# {para.split('.')[0]}")
                formatted.append(para)
            elif i % 3 == 0 and i > 0:
                # Add subheading every 3 paragraphs
                formatted.append(f"\n## {para.split('.')[0]}\n")
                formatted.append(para)
            else:
                formatted.append(para)
        
        return '\n\n'.join(formatted)
    
    def _format_code(self, code: str, language: str) -> str:
        """Format code with syntax highlighting"""
        return f"```{language}\n{code}\n```"

# Quality Checker
class QualityChecker:
    """Check quality of generated content"""
    
    def __init__(self):
        self.min_proposal_length = 150
        self.max_proposal_length = 500
        self.required_proposal_sections = [
            'greeting', 'understanding', 'experience', 'approach', 'timeline', 'pricing'
        ]
    
    def check_proposal(self, proposal: str) -> bool:
        """Check if proposal meets quality standards"""
        # Length check
        word_count = len(proposal.split())
        if word_count < self.min_proposal_length or word_count > self.max_proposal_length:
            return False
        
        # Content check
        proposal_lower = proposal.lower()
        
        # Check for key sections
        has_greeting = any(word in proposal_lower[:50] for word in ['hello', 'hi', 'dear', 'greetings'])
        has_understanding = any(word in proposal_lower for word in ['understand', 'project', 'requirements'])
        has_experience = any(word in proposal_lower for word in ['experience', 'worked', 'portfolio'])
        has_timeline = any(word in proposal_lower for word in ['deliver', 'timeline', 'days', 'weeks'])
        has_pricing = any(word in proposal_lower for word in ['price', 'cost', 'budget', 'rate'])
        
        return all([has_greeting, has_understanding, has_experience, has_timeline, has_pricing])
    
    def check_content(self, content: str, content_type: str) -> Dict[str, Any]:
        """Check content quality"""
        results = {
            'word_count': len(content.split()),
            'readability_score': self._calculate_readability(content),
            'keyword_density': self._calculate_keyword_density(content),
            'originality_score': self._check_originality(content),
            'grammar_errors': self._check_grammar(content)
        }
        
        # Overall quality score
        results['quality_score'] = (
            results['readability_score'] * 0.3 +
            results['originality_score'] * 0.4 +
            (100 - results['grammar_errors']) * 0.3
        )
        
        return results
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate Flesch Reading Ease score"""
        sentences = text.count('.') + text.count('!') + text.count('?')
        words = len(text.split())
        syllables = sum([self._count_syllables(word) for word in text.split()])
        
        if sentences == 0 or words == 0:
            return 0
        
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word"""
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        if word.endswith('le'):
            count += 1
        if count == 0:
            count += 1
        return count
    
    def _calculate_keyword_density(self, text: str, keywords: List[str] = None) -> float:
        """Calculate keyword density"""
        if not keywords:
            return 100.0
        
        text_lower = text.lower()
        word_count = len(text.split())
        keyword_count = sum(text_lower.count(keyword.lower()) for keyword in keywords)
        
        density = (keyword_count / word_count) * 100
        # Optimal density is 1-3%
        if 1 <= density <= 3:
            return 100.0
        elif density < 1:
            return density * 100
        else:
            return max(0, 100 - (density - 3) * 20)
    
    def _check_originality(self, text: str) -> float:
        """Check text originality (placeholder)"""
        # In production, integrate with plagiarism checker API
        return 95.0
    
    def _check_grammar(self, text: str) -> int:
        """Check grammar errors (placeholder)"""
        # In production, integrate with grammar checker API
        return 2

# Main Orchestrator
class FreelanceAutomationOrchestrator:
    """Main orchestration engine"""
    
    def __init__(self):
        self.config = Config()
        self.security = SecurityManager(self.config)
        self.cache = CacheManager(self.config)
        self.ai_manager = AIModelManager(self.config)
        self.work_engine = WorkGenerationEngine(self.ai_manager, self.config)
        self.web_automation = WebAutomationManager(self.config)
        
        # Initialize database
        self.db_engine = create_engine(
            self.config.get('database.url'),
            pool_size=self.config.get('database.pool_size', 10),
            max_overflow=self.config.get('database.max_overflow', 20),
            poolclass=QueuePool
        )
        Base.metadata.create_all(self.db_engine)
        self.Session = scoped_session(sessionmaker(bind=self.db_engine))
        
        # Platform clients
        self.platform_clients = {}
        
        # Job processing
        self.job_queue = asyncio.Queue()
        self.active_projects = {}
        self.running = False
        
        # Celery for background tasks
        self.celery = Celery(
            'freelance_automation',
            broker=self.config.get('redis.url'),
            backend=self.config.get('redis.url')
        )
        
        logger.info("Freelance Automation Orchestrator initialized")
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize platform clients
            for platform in ['upwork', 'fiverr', 'freelancer']:
                if self.config.get(f'platforms.{platform}.enabled'):
                    if platform == 'upwork':
                        client = UpworkClient(self.config, self.security)
                    # Add other platform clients
                    
                    self.platform_clients[platform] = client
            
            logger.info("Initialization complete")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise
    
    async def start(self):
        """Start automation system"""
        self.running = True
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._job_scanner()),
            asyncio.create_task(self._job_processor()),
            asyncio.create_task(self._project_monitor()),
            asyncio.create_task(self._quality_assurance())
        ]
        
        logger.info("Automation system started")
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.running = False
            await self.shutdown()
    
    async def _job_scanner(self):
        """Continuously scan for new jobs"""
        while self.running:
            try:
                scan_interval = self.config.get('automation.scan_interval', 300)
                
                for platform_name, client in self.platform_clients.items():
                    async with client:
                        # Get user preferences
                        filters = self._get_job_filters()
                        
                        # Scan for jobs
                        jobs = await client.get_jobs(filters)
                        
                        # Filter and queue jobs
                        for job in jobs:
                            if await self._should_apply(job):
                                await self.job_queue.put((platform_name, job))
                                logger.info(f"Queued job: {job.title} on {platform_name}")
                
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                logger.error(f"Job scanner error: {e}")
                await asyncio.sleep(60)
    
    async def _job_processor(self):
        """Process queued jobs"""
        while self.running:
            try:
                # Get job from queue
                platform_name, job = await asyncio.wait_for(
                    self.job_queue.get(),
                    timeout=30
                )
                
                # Check if already applied
                session = self.Session()
                existing = session.query(Job).filter_by(id=job.id).first()
                if existing and existing.proposal_sent:
                    session.close()
                    continue
                
                # Generate proposal
                profile = self._get_user_profile()
                proposal = await self.work_engine.generate_proposal(job, profile)
                
                if proposal:
                    # Submit proposal
                    client = self.platform_clients[platform_name]
                    async with client:
                        success = await client.submit_proposal(job.id, proposal)
                        
                        if success:
                            # Update database
                            job.proposal_sent = True
                            job.status = 'applied'
                            session.merge(job)
                            session.commit()
                            
                            logger.info(f"Submitted proposal for: {job.title}")
                        else:
                            logger.error(f"Failed to submit proposal for: {job.title}")
                
                session.close()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Job processor error: {e}")
    
    async def _project_monitor(self):
        """Monitor active projects"""
        while self.running:
            try:
                # Check for new messages
                for platform_name, client in self.platform_clients.items():
                    async with client:
                        messages = await client.get_messages()
                        
                        for message in messages:
                            await self._handle_client_message(platform_name, message)
                
                # Check project deadlines
                session = self.Session()
                active_projects = session.query(Project).filter_by(status='in_progress').all()
                
                for project in active_projects:
                    if project.deadline and project.deadline < datetime.now() + timedelta(hours=24):
                        logger.warning(f"Project {project.id} deadline approaching!")
                
                session.close()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Project monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _quality_assurance(self):
        """Perform quality checks on deliverables"""
        while self.running:
            try:
                # Check completed work quality
                session = self.Session()
                pending_qa = session.query(Project).filter_by(status='pending_qa').all()
                
                for project in pending_qa:
                    deliverables = json.loads(project.deliverables or '{}')
                    
                    for deliverable_type, content in deliverables.items():
                        quality_results = self.work_engine.quality_checker.check_content(
                            content,
                            deliverable_type
                        )
                        
                        if quality_results['quality_score'] >= 80:
                            project.status = 'ready_to_submit'
                        else:
                            # Needs improvement
                            project.status = 'needs_revision'
                            logger.warning(f"Project {project.id} needs quality improvement")
                
                session.commit()
                session.close()
                
                await asyncio.sleep(600)  # Check every 10 minutes
                
            except Exception as e:
                logger.error(f"Quality assurance error: {e}")
                await asyncio.sleep(60)
    
    async def _should_apply(self, job: Job) -> bool:
        """Determine if should apply to job"""
        try:
            # Check budget
            min_budget = self.config.get('automation.min_budget', 50)
            if job.budget_max and job.budget_max < min_budget:
                return False
            
            # Check skill match
            job_skills = set(json.loads(job.skills or '[]'))
            my_skills = set(self.config.get('profile.skills', []))
            
            if not job_skills:
                # If no skills specified, check description
                job_text = f"{job.title} {job.description}".lower()
                skill_matches = sum(1 for skill in my_skills if skill.lower() in job_text)
                match_score = skill_matches / max(len(my_skills), 1)
            else:
                match_score = len(job_skills.intersection(my_skills)) / len(job_skills)
            
            min_match_score = self.config.get('automation.min_match_score', 0.7)
            
            return match_score >= min_match_score
            
        except Exception as e:
            logger.error(f"Error checking job fit: {e}")
            return False
    
    async def _handle_client_message(self, platform: str, message: Dict):
        """Handle incoming client messages"""
        try:
            # Determine message type
            if 'revision' in message.get('subject', '').lower():
                await self._handle_revision_request(platform, message)
            elif 'feedback' in message.get('subject', '').lower():
                await self._handle_feedback(platform, message)
            else:
                # Generate appropriate response
                response = self.ai_manager.generate_text(
                    f"Generate a professional response to this client message: {message.get('content')}",
                    model_type='content'
                )
                
                # Send response
                # Implementation depends on platform API
                
        except Exception as e:
            logger.error(f"Error handling client message: {e}")
    
    def _get_job_filters(self) -> Dict:
        """Get job search filters from config"""
        return {
            'skills': self.config.get('profile.skills', []),
            'min_budget': self.config.get('automation.min_budget', 50),
            'max_budget': self.config.get('automation.max_budget', 10000),
            'job_type': self.config.get('automation.job_type', 'both'),
            'experience_level': self.config.get('automation.experience_level', 'intermediate')
        }
    
    def _get_user_profile(self) -> Dict:
        """Get user profile for proposals"""
        return {
            'name': self.config.get('profile.name', 'Freelancer'),
            'skills': self.config.get('profile.skills', []),
            'experience_years': self.config.get('profile.experience_years', 5),
            'portfolio_summary': self.config.get('profile.portfolio_summary', ''),
            'hourly_rate': self.config.get('profile.hourly_rate', 50)
        }
    
    async def shutdown(self):
        """Cleanup and shutdown"""
        self.running = False
        
        # Close platform clients
        for client in self.platform_clients.values():
            try:
                await client.__aexit__(None, None, None)
            except:
                pass
        
        # Cleanup web automation
        self.web_automation.cleanup()
        
        # Close database connections
        self.Session.remove()
        self.db_engine.dispose()
        
        logger.info("Shutdown complete")

# Web Interface
class FreelanceAutomationUI:
    """Gradio-based web interface"""
    
    def __init__(self, orchestrator: FreelanceAutomationOrchestrator):
        self.orchestrator = orchestrator
        self.interface = self._create_interface()
    
    def _create_interface(self):
        """Create Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            font-family: 'Inter', sans-serif;
            max-width: 1400px;
            margin: 0 auto;
        }
        .tab-nav {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 4px;
        }
        .primary-btn {
            background-color: #007bff !important;
            color: white !important;
            font-weight: 600;
        }
        .success-box {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            border-radius: 4px;
            padding: 12px;
        }
        .metric-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }
        """
        
        with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="AI Freelance Automation") as interface:
            
            # Header
            gr.Markdown(
                """
                # 🤖 AI Freelance Automation System
                ### Automate your freelance workflow with AI-powered tools
                """
            )
            
            # Main tabs
            with gr.Tabs() as tabs:
                
                # Dashboard Tab
                with gr.Tab("📊 Dashboard", id=1):
                    with gr.Row():
                        with gr.Column(scale=1):
                            active_jobs_count = gr.Number(
                                label="Active Jobs",
                                value=0,
                                elem_classes=["metric-card"]
                            )
                        with gr.Column(scale=1):
                            total_earnings = gr.Number(
                                label="Total Earnings ($)",
                                value=0.00,
                                elem_classes=["metric-card"]
                            )
                        with gr.Column(scale=1):
                            success_rate = gr.Number(
                                label="Success Rate (%)",
                                value=0.0,
                                elem_classes=["metric-card"]
                            )
                        with gr.Column(scale=1):
                            avg_rating = gr.Number(
                                label="Average Rating",
                                value=0.0,
                                elem_classes=["metric-card"]
                            )
                    
                    gr.Markdown("### Recent Activity")
                    activity_log = gr.Dataframe(
                        headers=["Time", "Platform", "Action", "Status"],
                        datatype=["str", "str", "str", "str"],
                        row_count=10,
                        col_count=(4, "fixed")
                    )
                    
                    with gr.Row():
                        refresh_dashboard_btn = gr.Button(
                            "🔄 Refresh Dashboard",
                            variant="secondary"
                        )
                        export_data_btn = gr.Button(
                            "📥 Export Data",
                            variant="secondary"
                        )
                
                # Configuration Tab
                with gr.Tab("⚙️ Configuration", id=2):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Platform Settings")
                            
                            platform_tabs = gr.Tabs()
                            with platform_tabs:
                                with gr.Tab("Upwork"):
                                    upwork_enabled = gr.Checkbox(
                                        label="Enable Upwork Integration",
                                        value=True
                                    )
                                    upwork_client_id = gr.Textbox(
                                        label="Client ID",
                                        type="password"
                                    )
                                    upwork_client_secret = gr.Textbox(
                                        label="Client Secret",
                                        type="password"
                                    )
                                
                                with gr.Tab("Fiverr"):
                                    fiverr_enabled = gr.Checkbox(
                                        label="Enable Fiverr Integration",
                                        value=True
                                    )
                                    fiverr_username = gr.Textbox(label="Username")
                                    fiverr_password = gr.Textbox(
                                        label="Password",
                                        type="password"
                                    )
                                
                                with gr.Tab("Freelancer"):
                                    freelancer_enabled = gr.Checkbox(
                                        label="Enable Freelancer Integration",
                                        value=False
                                    )
                                    freelancer_api_key = gr.Textbox(
                                        label="API Key",
                                        type="password"
                                    )
                        
                        with gr.Column():
                            gr.Markdown("### Profile Settings")
                            
                            profile_name = gr.Textbox(
                                label="Professional Name",
                                value="AI Freelancer"
                            )
                            
                            skills = gr.Textbox(
                                label="Skills (comma-separated)",
                                value="Python, JavaScript, Content Writing, Web Development",
                                lines=2
                            )
                            
                            hourly_rate = gr.Slider(
                                label="Hourly Rate ($)",
                                minimum=10,
                                maximum=200,
                                value=50,
                                step=5
                            )
                            
                            min_budget = gr.Slider(
                                label="Minimum Project Budget ($)",
                                minimum=10,
                                maximum=1000,
                                value=100,
                                step=10
                            )
                            
                            auto_apply = gr.Checkbox(
                                label="Auto-apply to matching jobs",
                                value=True
                            )
                            
                            portfolio_summary = gr.Textbox(
                                label="Portfolio Summary",
                                lines=4,
                                placeholder="Brief description of your experience and past projects..."
                            )
                    
                    save_config_btn = gr.Button(
                        "💾 Save Configuration",
                        variant="primary",
                        elem_classes=["primary-btn"]
                    )
                    
                    config_status = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes=["success-box"]
                    )
                
                # Job Scanner Tab
                with gr.Tab("🔍 Job Scanner", id=3):
                    with gr.Row():
                        scan_platform = gr.CheckboxGroup(
                            choices=["Upwork", "Fiverr", "Freelancer"],
                            value=["Upwork", "Fiverr"],
                            label="Platforms to Scan"
                        )
                        
                        job_type_filter = gr.Radio(
                            choices=["All", "Fixed Price", "Hourly"],
                            value="All",
                            label="Job Type"
                        )
                        
                        skill_filter = gr.Dropdown(
                            choices=["All Skills", "Python", "JavaScript", "Content Writing", "Web Design"],
                            value="All Skills",
                            label="Skill Filter",
                            multiselect=True
                        )
                    
                    with gr.Row():
                        scan_now_btn = gr.Button(
                            "🚀 Scan Now",
                            variant="primary",
                            elem_classes=["primary-btn"]
                        )
                        
                        auto_scan_toggle = gr.Checkbox(
                            label="Enable Auto-Scan (every 5 minutes)",
                            value=True
                        )
                    
                    gr.Markdown("### Available Jobs")
                    
                    jobs_table = gr.Dataframe(
                        headers=["Platform", "Title", "Budget", "Skills", "Match %", "Action"],
                        datatype=["str", "str", "str", "str", "number", "str"],
                        row_count=20,
                        interactive=False
                    )
                    
                    with gr.Row():
                        apply_selected_btn = gr.Button(
                            "✍️ Apply to Selected",
                            variant="secondary"
                        )
                        
                        view_details_btn = gr.Button(
                            "👁️ View Details",
                            variant="secondary"
                        )
                
                # Work Generation Tab
                with gr.Tab("✨ Work Generation", id=4):
                    with gr.Row():
                        with gr.Column():
                            work_type = gr.Dropdown(
                                choices=[
                                    "Blog Post",
                                    "Article",
                                    "Product Description",
                                    "Website Copy",
                                    "Python Script",
                                    "JavaScript Code",
                                    "eBook Chapter",
                                    "Email Campaign"
                                ],
                                value="Blog Post",
                                label="Content Type"
                            )
                            
                            topic_input = gr.Textbox(
                                label="Topic/Title",
                                placeholder="Enter the topic or title..."
                            )
                            
                            requirements_input = gr.Textbox(
                                label="Requirements",
                                lines=5,
                                placeholder="Describe the specific requirements..."
                            )
                            
                            word_count = gr.Slider(
                                label="Word Count",
                                minimum=100,
                                maximum=5000,
                                value=1000,
                                step=100
                            )
                            
                            tone_style = gr.Radio(
                                choices=["Professional", "Casual", "Academic", "Creative"],
                                value="Professional",
                                label="Tone/Style"
                            )
                            
                            keywords_input = gr.Textbox(
                                label="Keywords (comma-separated)",
                                placeholder="SEO keywords to include..."
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Generated Output")
                            
                            output_display = gr.Textbox(
                                label="Preview",
                                lines=15,
                                interactive=True
                            )
                            
                            quality_metrics = gr.JSON(
                                label="Quality Metrics"
                            )
                            
                            with gr.Row():
                                generate_btn = gr.Button(
                                    "🎨 Generate",
                                    variant="primary",
                                    elem_classes=["primary-btn"]
                                )
                                
                                humanize_btn = gr.Button(
                                    "🧑 Humanize",
                                    variant="secondary"
                                )
                                
                                export_btn = gr.Button(
                                    "📄 Export",
                                    variant="secondary"
                                )
                
                # Projects Tab
                with gr.Tab("📁 Projects", id=5):
                    gr.Markdown("### Active Projects")
                    
                    projects_table = gr.Dataframe(
                        headers=["ID", "Platform", "Client", "Title", "Status", "Deadline", "Progress"],
                        datatype=["str", "str", "str", "str", "str", "str", "number"],
                        row_count=10
                    )
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Project Details")
                            
                            selected_project = gr.Dropdown(
                                label="Select Project",
                                choices=[]
                            )
                            
                            project_status = gr.Textbox(
                                label="Status",
                                interactive=False
                            )
                            
                            project_deadline = gr.Textbox(
                                label="Deadline",
                                interactive=False
                            )
                            
                            client_messages = gr.Textbox(
                                label="Recent Messages",
                                lines=5,
                                interactive=False
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Deliverables")
                            
                            deliverable_type = gr.Dropdown(
                                label="Type",
                                choices=["Document", "Code", "Design", "Other"]
                            )
                            
                            deliverable_file = gr.File(
                                label="Upload Deliverable"
                            )
                            
                            deliverable_notes = gr.Textbox(
                                label="Notes",
                                lines=3
                            )
                            
                            submit_deliverable_btn = gr.Button(
                                "📤 Submit Deliverable",
                                variant="primary"
                            )
                
                # Analytics Tab
                with gr.Tab("📈 Analytics", id=6):
                    with gr.Row():
                        date_range = gr.Dropdown(
                            choices=["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"],
                            value="Last 30 Days",
                            label="Date Range"
                        )
                        
                        platform_filter = gr.CheckboxGroup(
                            choices=["Upwork", "Fiverr", "Freelancer"],
                            value=["Upwork", "Fiverr"],
                            label="Platforms"
                        )
                    
                    with gr.Row():
                        earnings_chart = gr.Plot(label="Earnings Over Time")
                        success_rate_chart = gr.Plot(label="Success Rate by Platform")
                    
                    with gr.Row():
                        skill_performance = gr.Plot(label="Performance by Skill")
                        client_satisfaction = gr.Plot(label="Client Satisfaction")
                    
                    gr.Markdown("### Detailed Statistics")
                    
                    stats_table = gr.Dataframe(
                        headers=["Metric", "Value", "Change", "Trend"],
                        datatype=["str", "str", "str", "str"],
                        row_count=10
                    )
                
                # Settings Tab
                with gr.Tab("🔧 Settings", id=7):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### AI Model Settings")
                            
                            content_model = gr.Dropdown(
                                label="Content Generation Model",
                                choices=["mistral", "llama2", "gpt-neo"],
                                value="mistral"
                            )
                            
                            code_model = gr.Dropdown(
                                label="Code Generation Model",
                                choices=["codellama", "starcoder", "codegen"],
                                value="codellama"
                            )
                            
                            temperature = gr.Slider(
                                label="Generation Temperature",
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.1
                            )
                            
                            max_tokens = gr.Slider(
                                label="Max Tokens",
                                minimum=100,
                                maximum=4000,
                                value=2000,
                                step=100
                            )
                        
                        with gr.Column():
                            gr.Markdown("### Automation Settings")
                            
                            scan_interval = gr.Slider(
                                label="Job Scan Interval (minutes)",
                                minimum=1,
                                maximum=60,
                                value=5,
                                step=1
                            )
                            
                            max_concurrent = gr.Slider(
                                label="Max Concurrent Jobs",
                                minimum=1,
                                maximum=10,
                                value=5,
                                step=1
                            )
                            
                            auto_humanize = gr.Checkbox(
                                label="Auto-humanize all content",
                                value=True
                            )
                            
                            quality_threshold = gr.Slider(
                                label="Quality Threshold (%)",
                                minimum=50,
                                maximum=100,
                                value=80,
                                step=5
                            )
                    
                    gr.Markdown("### Notification Settings")
                    
                    with gr.Row():
                        email_notifications = gr.Checkbox(
                            label="Email Notifications",
                            value=True
                        )
                        
                        notification_email = gr.Textbox(
                            label="Email Address",
                            placeholder="your@email.com"
                        )
                        
                        notification_events = gr.CheckboxGroup(
                            label="Notify me when:",
                            choices=[
                                "New job matches criteria",
                                "Proposal accepted",
                                "Client sends message",
                                "Project deadline approaching",
                                "Payment received"
                            ],
                            value=["Proposal accepted", "Client sends message"]
                        )
                    
                    save_settings_btn = gr.Button(
                        "💾 Save Settings",
                        variant="primary",
                        elem_classes=["primary-btn"]
                    )
            
            # Event Handlers
            def refresh_dashboard():
                # Get real-time statistics
                stats = self._get_dashboard_stats()
                activity = self._get_recent_activity()
                
                return (
                    stats['active_jobs'],
                    stats['total_earnings'],
                    stats['success_rate'],
                    stats['avg_rating'],
                    activity
                )
            
            def save_configuration(*args):
                # Save configuration to file
                try:
                    # Update config
                    self._update_configuration(args)
                    return "✅ Configuration saved successfully!"
                except Exception as e:
                    return f"❌ Error saving configuration: {str(e)}"
            
            def scan_jobs(platforms, job_type, skills):
                # Trigger job scan
                jobs = asyncio.run(self._scan_jobs_async(platforms, job_type, skills))
                return jobs
            
            def generate_content(work_type, topic, requirements, word_count, tone, keywords):
                # Generate content
                result = asyncio.run(self._generate_content_async(
                    work_type, topic, requirements, word_count, tone, keywords
                ))
                return result['content'], result['metrics']
            
            # Connect event handlers
            refresh_dashboard_btn.click(
                fn=refresh_dashboard,
                outputs=[
                    active_jobs_count,
                    total_earnings,
                    success_rate,
                    avg_rating,
                    activity_log
                ]
            )
            
            save_config_btn.click(
                fn=save_configuration,
                inputs=[
                    upwork_enabled, upwork_client_id, upwork_client_secret,
                    fiverr_enabled, fiverr_username, fiverr_password,
                    freelancer_enabled, freelancer_api_key,
                    profile_name, skills, hourly_rate, min_budget,
                    auto_apply, portfolio_summary
                ],
                outputs=[config_status]
            )
            
            scan_now_btn.click(
                fn=scan_jobs,
                inputs=[scan_platform, job_type_filter, skill_filter],
                outputs=[jobs_table]
            )
            
            generate_btn.click(
                fn=generate_content,
                inputs=[
                    work_type, topic_input, requirements_input,
                    word_count, tone_style, keywords_input
                ],
                outputs=[output_display, quality_metrics]
            )
            
            # Auto-refresh dashboard on load
            interface.load(
                fn=refresh_dashboard,
                outputs=[
                    active_jobs_count,
                    total_earnings,
                    success_rate,
                    avg_rating,
                    activity_log
                ]
            )
        
        return interface
    
    def _get_dashboard_stats(self) -> Dict:
        """Get dashboard statistics"""
        session = self.orchestrator.Session()
        
        try:
            # Active jobs
            active_jobs = session.query(Project).filter_by(status='in_progress').count()
            
            # Total earnings
            total_earnings = session.query(
                func.sum(Project.earnings)
            ).filter_by(status='completed').scalar() or 0
            
            # Success rate
            total_projects = session.query(Project).count()
            successful_projects = session.query(Project).filter_by(status='completed').count()
            success_rate = (successful_projects / total_projects * 100) if total_projects > 0 else 0
            
            # Average rating
            avg_rating = session.query(
                func.avg(Project.rating)
            ).filter(Project.rating.isnot(None)).scalar() or 0
            
            return {
                'active_jobs': active_jobs,
                'total_earnings': total_earnings,
                'success_rate': round(success_rate, 1),
                'avg_rating': round(avg_rating, 1)
            }
            
        finally:
            session.close()
    
    def _get_recent_activity(self) -> List[List[str]]:
        """Get recent activity log"""
        # This would fetch from a logging table in production
        return [
            [datetime.now().strftime("%Y-%m-%d %H:%M"), "Upwork", "Proposal Sent", "Success"],
            [datetime.now().strftime("%Y-%m-%d %H:%M"), "Fiverr", "Job Completed", "Success"],
            # Add more activities
        ]
    
    async def _scan_jobs_async(self, platforms, job_type, skills):
        """Scan jobs asynchronously"""
        # Implement job scanning logic
        return [
            ["Upwork", "Python Developer Needed", "$500-1000", "Python, Django", 85, "Apply"],
            ["Fiverr", "Content Writer for Tech Blog", "$100-200", "Writing, SEO", 92, "Apply"],
            # Add more jobs
        ]
    
    async def _generate_content_async(self, work_type, topic, requirements, word_count, tone, keywords):
        """Generate content asynchronously"""
        content = await self.orchestrator.work_engine.generate_content(
            work_type.lower().replace(" ", "_"),
            {
                'topic': topic,
                'requirements': requirements,
                'word_count': word_count,
                'style': tone,
                'keywords': keywords.split(',') if keywords else []
            }
        )
        
        # Check quality
        quality_metrics = self.orchestrator.work_engine.quality_checker.check_content(
            content.get('content', ''),
            work_type
        )
        
        return {
            'content': content.get('content', ''),
            'metrics': quality_metrics
        }
    
    def launch(self, **kwargs):
        """Launch the web interface"""
        self.interface.launch(**kwargs)

# Main entry point
async def main():
    """Main application entry point"""
    try:
        # Initialize orchestrator
        orchestrator = FreelanceAutomationOrchestrator()
        await orchestrator.initialize()
        
        # Create UI
        ui = FreelanceAutomationUI(orchestrator)
        
        # Run both orchestrator and UI
        await asyncio.gather(
            orchestrator.start(),
            asyncio.to_thread(ui.launch, server_name="0.0.0.0", server_port=7860, share=False)
        )
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise
    finally:
        await orchestrator.shutdown()

if __name__ == "__main__":
    # Run the application
    asyncio.run(main())
