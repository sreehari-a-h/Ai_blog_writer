#!/usr/bin/env python3
"""
SEO Blog Generator API - A complete Ubersuggest alternative focused on blog content generation
Integrates keyword discovery, clustering, outline generation, and AI writing
"""

import asyncio
import json
import re
import requests
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import openai
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import hashlib
import time
from urllib.parse import quote_plus
import aiohttp
from bs4 import BeautifulSoup
import logging
import os
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SEO Blog Generator API",
    description="Complete SEO blog content generation pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global configurations
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
MAX_KEYWORDS_PER_REQUEST = 50
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Replace with actual key
SERP_API_KEY = os.getenv("SERP_API_KEY")  # Replace with actual SerpAPI key

# Initialize models (lazy loading)
sentence_model = None
openai.api_key = OPENAI_API_KEY

def get_sentence_model():
    global sentence_model
    if sentence_model is None:
        sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    return sentence_model

# Database setup
def init_db():
    conn = sqlite3.connect('seo_blog_cache.db')
    cursor = conn.cursor()
    
    # Keywords cache table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS keyword_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed_keyword TEXT NOT NULL,
            keywords_json TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source TEXT DEFAULT 'google_suggest'
        )
    ''')
    
    # Blog content cache
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS blog_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            topic TEXT NOT NULL,
            outline_json TEXT,
            blog_content TEXT,
            meta_title TEXT,
            meta_description TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Content calendar
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS content_calendar (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            target_keyword TEXT NOT NULL,
            scheduled_date DATE,
            status TEXT DEFAULT 'planned',
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

init_db()

# Pydantic models
class KeywordRequest(BaseModel):
    seed_keywords: List[str]
    language: str = "en"
    country: str = "US"
    limit: int = 20

class BlogTopicRequest(BaseModel):
    keywords: List[str]
    target_audience: str = "general"
    content_type: str = "informational"
    min_cluster_size: int = 3

class BlogOutlineRequest(BaseModel):
    main_topic: str
    target_keywords: List[str]
    audience: str = "general"
    content_length: str = "medium"  # short, medium, long

class BlogGenerationRequest(BaseModel):
    topic: str
    outline: Dict[str, Any]
    target_keywords: List[str]
    audience: str = "general"
    tone: str = "professional"  # professional, casual, authoritative
    word_count: int = 1200

class ContentCalendarRequest(BaseModel):
    topics: List[str]
    start_date: str
    frequency: str = "weekly"  # daily, weekly, bi-weekly, monthly

# API Integration Classes
class GoogleSuggestAPI:
    """Fetch keyword suggestions from Google Suggest API"""
    
    @staticmethod
    async def get_suggestions(seed_keyword: str, language: str = "en") -> List[str]:
        suggestions = []
        
        # Multiple suggestion sources
        suggestion_queries = [
            f"{seed_keyword}",
            f"best {seed_keyword}",
            f"how to {seed_keyword}",
            f"{seed_keyword} guide",
            f"{seed_keyword} tips",
            f"{seed_keyword} vs",
            f"{seed_keyword} for beginners",
            f"{seed_keyword} 2024",
            f"why {seed_keyword}",
            f"{seed_keyword} reviews"
        ]
        
        async with aiohttp.ClientSession() as session:
            for query in suggestion_queries:
                try:
                    url = f"https://suggestqueries.google.com/complete/search"
                    params = {
                        'client': 'chrome',
                        'q': query,
                        'hl': language
                    }
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.text()
                            # Parse Google Suggest response
                            try:
                                parsed = json.loads(data)
                                if len(parsed) > 1 and isinstance(parsed[1], list):
                                    suggestions.extend(parsed[1])
                            except json.JSONDecodeError:
                                pass
                    
                    await asyncio.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logger.warning(f"Error fetching suggestions for {query}: {e}")
                    continue
        
        # Clean and deduplicate
        cleaned_suggestions = []
        seen = set()
        
        for suggestion in suggestions:
            if isinstance(suggestion, str):
                suggestion = suggestion.lower().strip()
                if suggestion and suggestion not in seen and len(suggestion) > 3:
                    cleaned_suggestions.append(suggestion)
                    seen.add(suggestion)
        
        return cleaned_suggestions[:50]  # Limit results

class SerpAPI:
    """Fetch 'People Also Ask' and related searches using SerpAPI"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
    
    async def get_related_questions(self, keyword: str) -> List[str]:
        if not self.api_key or self.api_key == os.getenv("SERP_API_KEY"):
            # Fallback: generate mock related questions
            return self._generate_mock_questions(keyword)
        
        params = {
            'q': keyword,
            'engine': 'google',
            'api_key': self.api_key,
            'num': 10
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        questions = []
                        # Extract People Also Ask
                        if 'related_questions' in data:
                            for q in data['related_questions']:
                                if 'question' in q:
                                    questions.append(q['question'])
                        
                        # Extract related searches
                        if 'related_searches' in data:
                            for search in data['related_searches']:
                                if 'query' in search:
                                    questions.append(search['query'])
                        
                        return questions[:15]
        except Exception as e:
            logger.warning(f"SerpAPI error: {e}")
            return self._generate_mock_questions(keyword)
        
        return []
    
    def _generate_mock_questions(self, keyword: str) -> List[str]:
        """Generate mock questions when API is not available"""
        question_templates = [
            f"What is {keyword}?",
            f"How does {keyword} work?",
            f"Why is {keyword} important?",
            f"What are the benefits of {keyword}?",
            f"How to choose the best {keyword}?",
            f"What are the types of {keyword}?",
            f"How much does {keyword} cost?",
            f"What are alternatives to {keyword}?",
            f"Is {keyword} worth it?",
            f"How to get started with {keyword}?"
        ]
        return question_templates

class KeywordClusterer:
    """Cluster keywords using semantic similarity"""
    
    def __init__(self):
        self.model = get_sentence_model()
    
    def cluster_keywords(self, keywords: List[str], min_cluster_size: int = 3, max_clusters: int = 10) -> Dict[str, List[str]]:
        if len(keywords) < min_cluster_size:
            return {"general": keywords}
        
        # Generate embeddings
        embeddings = self.model.encode(keywords)
        
        # Determine optimal number of clusters
        n_clusters = min(max_clusters, len(keywords) // min_cluster_size)
        n_clusters = max(2, n_clusters)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Group keywords by cluster
        clusters = {}
        for i, keyword in enumerate(keywords):
            cluster_id = cluster_labels[i]
            cluster_name = f"topic_{cluster_id}"
            
            if cluster_name not in clusters:
                clusters[cluster_name] = []
            clusters[cluster_name].append(keyword)
        
        # Name clusters based on most common terms
        named_clusters = {}
        for cluster_name, cluster_keywords in clusters.items():
            if len(cluster_keywords) >= min_cluster_size:
                # Extract common words to name cluster
                all_words = ' '.join(cluster_keywords).lower().split()
                word_freq = {}
                for word in all_words:
                    if len(word) > 3:  # Filter short words
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                if word_freq:
                    top_word = max(word_freq, key=word_freq.get)
                    named_clusters[f"{top_word}_guide"] = cluster_keywords
                else:
                    named_clusters[cluster_name] = cluster_keywords
        
        return named_clusters if named_clusters else {"general": keywords}

class SEOOutlineGenerator:
    """Generate SEO-optimized blog outlines"""
    
    def __init__(self):
        self.serp_api = SerpAPI(SERP_API_KEY)
    
    async def generate_outline(self, main_topic: str, target_keywords: List[str], audience: str = "general") -> Dict[str, Any]:
        # Get related questions
        related_questions = await self.serp_api.get_related_questions(main_topic)
        
        # Generate outline structure
        outline = {
            "meta_title": self._generate_meta_title(main_topic, target_keywords),
            "meta_description": self._generate_meta_description(main_topic, target_keywords),
            "h1": main_topic.title(),
            "introduction": {
                "hook": f"Discover everything you need to know about {main_topic}",
                "keywords_to_include": target_keywords[:3]
            },
            "main_sections": [],
            "faq": related_questions[:8],
            "conclusion": {
                "summary": f"Key takeaways about {main_topic}",
                "cta": "Ready to get started?"
            }
        }
        
        # Generate main sections based on keyword analysis
        sections = self._generate_sections(main_topic, target_keywords, audience)
        outline["main_sections"] = sections
        
        return outline
    
    def _generate_meta_title(self, topic: str, keywords: List[str]) -> str:
        """Generate SEO meta title (max 60 chars)"""
        main_keyword = keywords[0] if keywords else topic
        year = datetime.now().year
        
        templates = [
            f"{main_keyword.title()} Guide {year}",
            f"Complete {main_keyword.title()} Guide",
            f"Best {main_keyword.title()} Tips & Guide",
            f"How to Master {main_keyword.title()}",
            f"Ultimate {main_keyword.title()} Guide"
        ]
        
        for template in templates:
            if len(template) <= 60:
                return template
        
        return f"{main_keyword.title()}"[:60]
    
    def _generate_meta_description(self, topic: str, keywords: List[str]) -> str:
        """Generate SEO meta description (max 160 chars)"""
        main_keyword = keywords[0] if keywords else topic
        
        description = f"Learn everything about {main_keyword}. Complete guide with tips, best practices, and expert insights. Get started today!"
        
        return description[:160]
    
    def _generate_sections(self, topic: str, keywords: List[str], audience: str) -> List[Dict[str, Any]]:
        """Generate main content sections"""
        sections = [
            {
                "h2": f"What is {topic.title()}?",
                "subsections": [
                    f"Definition and basics",
                    f"Why {topic} matters",
                    f"Key benefits"
                ],
                "keywords_to_include": keywords[:2]
            },
            {
                "h2": f"Types of {topic.title()}",
                "subsections": [
                    f"Popular {topic} categories",
                    f"Choosing the right type",
                    f"Comparison guide"
                ],
                "keywords_to_include": [kw for kw in keywords if "types" in kw or "best" in kw][:2]
            },
            {
                "h2": f"How to Get Started with {topic.title()}",
                "subsections": [
                    f"Step-by-step guide",
                    f"Common mistakes to avoid",
                    f"Pro tips for beginners"
                ],
                "keywords_to_include": [kw for kw in keywords if "how to" in kw or "guide" in kw][:2]
            },
            {
                "h2": f"Advanced {topic.title()} Strategies",
                "subsections": [
                    f"Expert techniques",
                    f"Advanced tips",
                    f"Optimization strategies"
                ],
                "keywords_to_include": [kw for kw in keywords if "advanced" in kw or "tips" in kw][:2]
            }
        ]
        
        return sections

class AIBlogWriter:
    """Generate blog content using Hugging Face OpenAI-compatible API"""

    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",    
            api_key=os.getenv("HF_TOKEN")  # expects HF_TOKEN in env
        )

    async def generate_blog(self, topic, outline, target_keywords,
                            audience="general", tone="professional", word_count=1200):

        prompt = self._create_blog_prompt(topic, outline, target_keywords, audience, tone, word_count)

        try:
            # Using Hugging Face's OpenAI-compatible chat API
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="Qwen/Qwen3-Next-80B-A3B-Instruct:novita",  # choose any HF-supported model
                messages=[
                    {"role": "system", "content": "You are an expert SEO blog writer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=min(4000, word_count * 2),
            )

            content = response.choices[0].message.content.strip()
            return {
                "content": content,
                "meta_title": outline.get("meta_title", ""),
                "meta_description": outline.get("meta_description", ""),
                "word_count": len(content.split()),
                "headings": self._extract_headings(content)
            }

        except Exception as e:
            logger.error(f"Hugging Face API error: {e}")
            return self._generate_mock_blog(topic, outline, target_keywords, word_count)

    
    def _create_blog_prompt(self, topic: str, outline: Dict[str, Any], keywords: List[str], 
                           audience: str, tone: str, word_count: int) -> str:
        
        keywords_str = ", ".join(keywords[:10])
        sections = "\n".join([f"- {section['h2']}" for section in outline.get('main_sections', [])])
        faqs = "\n".join([f"- {faq}" for faq in outline.get('faq', [])[:5]])
        
        prompt = f"""
Write a comprehensive, SEO-optimized blog article with the following specifications:

TOPIC: {topic}
TARGET AUDIENCE: {audience}
TONE: {tone}
WORD COUNT: Approximately {word_count} words

KEYWORDS TO NATURALLY INCLUDE: {keywords_str}

REQUIRED STRUCTURE:
{sections}

FAQS TO ADDRESS:
{faqs}

REQUIREMENTS:
- Start with an engaging introduction that hooks the reader
- Use clear, scannable headings (H2, H3)
- Include actionable tips and insights
- Naturally incorporate target keywords (avoid keyword stuffing)
- Write in {tone} tone suitable for {audience} audience
- End with a compelling conclusion and call-to-action
- Include FAQ section addressing common questions
- Ensure content is informative, engaging, and valuable

OUTPUT FORMAT:
Return the complete blog post with proper markdown formatting for headings.
"""
        return prompt
    
    async def _call_openai_async(self, prompt: str, word_count: int) -> str:
        """Make async call to OpenAI API"""
        try:
            response = await asyncio.to_thread(
                self.client.completions.create,
                model="gpt-3.5-turbo-instruct",
                prompt=prompt,
                max_tokens=min(4000, word_count * 2),
                temperature=0.7,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            return response.choices[0].text.strip()
        except Exception as e:
            logger.error(f"OpenAI async call failed: {e}")
            raise
    
    def _parse_blog_response(self, response: str, outline: Dict[str, Any]) -> Dict[str, str]:
        """Parse the AI response into structured blog data"""
        return {
            "content": response,
            "meta_title": outline.get("meta_title", ""),
            "meta_description": outline.get("meta_description", ""),
            "word_count": len(response.split()),
            "headings": self._extract_headings(response)
        }
    
    def _extract_headings(self, content: str) -> List[str]:
        """Extract headings from markdown content"""
        headings = []
        lines = content.split('\n')
        for line in lines:
            if line.startswith('#'):
                headings.append(line.strip())
        return headings
    
    def _generate_mock_blog(self, topic: str, outline: Dict[str, Any], keywords: List[str], word_count: int) -> Dict[str, str]:
        """Generate a mock blog when OpenAI API is not available"""
        
        mock_content = f"""# {outline.get('h1', topic.title())}

## Introduction

{outline['introduction']['hook'] if 'introduction' in outline else f'Welcome to our comprehensive guide on {topic}'}. In this article, we'll explore everything you need to know about {topic}, including best practices, tips, and actionable insights.

## What is {topic.title()}?

{topic.title()} is an essential concept that can significantly impact your success. Understanding the fundamentals is crucial for anyone looking to master this topic.

### Key Benefits

- Improved efficiency and results
- Better understanding of core concepts  
- Practical application of knowledge
- Enhanced decision-making capabilities

## Getting Started with {topic.title()}

To begin your journey with {topic}, follow these essential steps:

1. **Research and Planning**: Start by understanding your goals and requirements
2. **Choose the Right Approach**: Select the method that best fits your needs
3. **Implementation**: Put your plan into action with careful execution
4. **Monitor and Optimize**: Track your progress and make improvements

## Advanced Strategies

Once you've mastered the basics, consider these advanced techniques:

- **Optimization Methods**: Fine-tune your approach for better results
- **Integration Techniques**: Combine multiple strategies effectively
- **Scaling Considerations**: Expand your efforts while maintaining quality

## Frequently Asked Questions

{self._generate_mock_faqs(topic, outline.get('faq', []))}

## Conclusion

{topic.title()} offers tremendous opportunities for those willing to invest time and effort in learning. By following the strategies outlined in this guide, you'll be well-equipped to succeed.

Ready to get started? Begin implementing these strategies today and see the difference they can make!
"""
        
        return {
            "content": mock_content,
            "meta_title": outline.get("meta_title", f"{topic.title()} - Complete Guide"),
            "meta_description": outline.get("meta_description", f"Learn everything about {topic} with our comprehensive guide."),
            "word_count": len(mock_content.split()),
            "headings": self._extract_headings(mock_content)
        }
    
    def _generate_mock_faqs(self, topic: str, faqs: List[str]) -> str:
        """Generate mock FAQ section"""
        if not faqs:
            faqs = [
                f"What is {topic}?",
                f"How do I get started with {topic}?",
                f"What are the benefits of {topic}?",
                f"How much does {topic} cost?",
                f"Is {topic} suitable for beginners?"
            ]
        
        faq_content = ""
        for i, faq in enumerate(faqs[:5]):
            faq_content += f"\n### {faq}\n\nThis is a detailed answer to the question about {topic}. We provide comprehensive information to help you understand this topic better.\n"
        
        return faq_content

# API Endpoints

@app.get("/")
async def root():
    return {
        "message": "SEO Blog Generator API",
        "version": "1.0.0",
        "endpoints": {
            "keywords": "/api/keywords",
            "topics": "/api/blog-topics", 
            "outline": "/api/blog-outline",
            "generate": "/api/generate-blog",
            "calendar": "/api/content-calendar"
        }
    }

@app.post("/api/keywords")
async def get_keywords(request: KeywordRequest):
    """Fetch keyword suggestions for multiple seed keywords"""
    try:
        google_api = GoogleSuggestAPI()
        all_keywords = []

        for seed in request.seed_keywords:
            # --- check cache ---
            conn = sqlite3.connect('seo_blog_cache.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT keywords_json FROM keyword_cache 
                WHERE seed_keyword = ? AND timestamp > datetime('now', '-1 day')
                ORDER BY timestamp DESC LIMIT 1
            ''', (seed,))
            cached = cursor.fetchone()
            if cached:
                keywords = json.loads(cached[0])
            else:
                keywords = await google_api.get_suggestions(seed, request.language)
                keywords = keywords[:request.limit]
                cursor.execute('''
                    INSERT INTO keyword_cache (seed_keyword, keywords_json, source)
                    VALUES (?, ?, ?)
                ''', (seed, json.dumps(keywords), "google_suggest"))
                conn.commit()
            conn.close()

            all_keywords.extend(keywords)

        # Deduplicate across all seed keywords
        unique_keywords = list(dict.fromkeys(all_keywords))

        return {
            "seed_keywords": request.seed_keywords,
            "keywords": unique_keywords,
            "count": len(unique_keywords),
            "source": "mixed"
        }

    except Exception as e:
        logger.error(f"Error fetching keywords: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching keywords: {str(e)}")


@app.post("/api/blog-topics")
async def generate_blog_topics(request: BlogTopicRequest):
    """Cluster keywords into blog topic suggestions"""
    try:
        clusterer = KeywordClusterer()
        clusters = clusterer.cluster_keywords(request.keywords, request.min_cluster_size)
        
        # Generate blog topic suggestions
        blog_topics = []
        for cluster_name, cluster_keywords in clusters.items():
            if len(cluster_keywords) >= request.min_cluster_size:
                # Generate topic title
                main_keyword = cluster_keywords[0]
                topic_title = f"Complete Guide to {main_keyword.title()}"
                
                blog_topics.append({
                    "topic_title": topic_title,
                    "main_keyword": main_keyword,
                    "related_keywords": cluster_keywords,
                    "estimated_difficulty": "medium",  # You could implement difficulty scoring
                    "content_type": request.content_type,
                    "target_audience": request.target_audience
                })
        
        return {
            "total_topics": len(blog_topics),
            "blog_topics": blog_topics,
            "clustering_summary": {
                "original_keywords": len(request.keywords),
                "clusters_found": len(clusters),
                "viable_topics": len(blog_topics)
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating blog topics: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating topics: {str(e)}")

@app.post("/api/blog-outline")
async def generate_blog_outline(request: BlogOutlineRequest):
    """Generate SEO-optimized blog outline"""
    try:
        outline_generator = SEOOutlineGenerator()
        outline = await outline_generator.generate_outline(
            request.main_topic,
            request.target_keywords,
            request.audience
        )
        
        return {
            "topic": request.main_topic,
            "outline": outline,
            "target_keywords": request.target_keywords,
            "estimated_word_count": 1200 if request.content_length == "medium" else (800 if request.content_length == "short" else 2000),
            "seo_score": 85  # Mock SEO score - you could implement real scoring
        }
        
    except Exception as e:
        logger.error(f"Error generating outline: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating outline: {str(e)}")

@app.post("/api/generate-blog")
async def generate_blog_content(request: BlogGenerationRequest):
    """Generate complete SEO blog content"""
    try:
        # Create content hash for caching
        content_data = f"{request.topic}_{json.dumps(request.target_keywords)}_{request.audience}_{request.word_count}"
        content_hash = hashlib.md5(content_data.encode()).hexdigest()
        
        # Check cache
        conn = sqlite3.connect('seo_blog_cache.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT blog_content, meta_title, meta_description FROM blog_cache
            WHERE content_hash = ?
        ''', (content_hash,))
        
        cached = cursor.fetchone()
        if cached:
            conn.close()
            return {
                "topic": request.topic,
                "content": cached[0],
                "meta_title": cached[1],
                "meta_description": cached[2],
                "source": "cache",
                "generated_at": datetime.now().isoformat()
            }
        
        # Generate fresh content
        blog_writer = AIBlogWriter()
        result = await blog_writer.generate_blog(
            request.topic,
            request.outline,
            request.target_keywords,
            request.audience,
            request.tone,
            request.word_count
        )
        
        # Cache the result
        cursor.execute('''
            INSERT OR REPLACE INTO blog_cache 
            (content_hash, topic, outline_json, blog_content, meta_title, meta_description)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            content_hash,
            request.topic,
            json.dumps(request.outline),
            result["content"],
            result["meta_title"],
            result["meta_description"]
        ))
        
        conn.commit()
        conn.close()
        
        return {
            "topic": request.topic,
            "content": result["content"],
            "meta_title": result["meta_title"],
            "meta_description": result["meta_description"],
            "word_count": result["word_count"],
            "headings": result["headings"],
            "source": "fresh",
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating blog: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating blog: {str(e)}")

@app.post("/api/content-calendar")
async def generate_content_calendar(request: ContentCalendarRequest):
    """Generate content calendar based on topics"""
    try:
        calendar_items = []
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        
        # Calculate date intervals
        interval_days = {
            "daily": 1,
            "weekly": 7,
            "bi-weekly": 14,
            "monthly": 30
        }
        
        days_interval = interval_days.get(request.frequency, 7)
        
        # Generate calendar entries
        for i, topic in enumerate(request.topics):
            scheduled_date = start_date + timedelta(days=i * days_interval)
            
            calendar_items.append({
                "topic": topic,
                "scheduled_date": scheduled_date.strftime("%Y-%m-%d"),
                "status": "planned",
                "priority": "medium",
                "estimated_work_hours": 4
            })
        
        # Save to database
        conn = sqlite3.connect('seo_blog_cache.db')
        cursor = conn.cursor()
        
        for item in calendar_items:
            cursor.execute('''
                INSERT INTO content_calendar (topic, target_keyword, scheduled_date, status)
                VALUES (?, ?, ?, ?)
            ''', (item["topic"], item["topic"], item["scheduled_date"], item["status"]))
        
        conn.commit()
        conn.close()
        
        return {
            "calendar": calendar_items,
            "summary": {
                "total_topics": len(request.topics),
                "frequency": request.frequency,
                "start_date": request.start_date,
                "end_date": calendar_items[-1]["scheduled_date"] if calendar_items else request.start_date
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating calendar: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating calendar: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "database": "connected",
            "sentence_transformer": "loaded" if sentence_model else "not_loaded",
            "apis": {
                "google_suggest": "active",
                "openai": "configured" if OPENAI_API_KEY != "your-openai-api-key-here" else "not_configured",
                "serp_api": "configured" if SERP_API_KEY == os.getenv("SERP_API_KEY") else "not_configured"
            }
        }
    }

@app.get("/api/stats")
async def get_stats():
    """Get usage statistics"""
    try:
        conn = sqlite3.connect('seo_blog_cache.db')
        cursor = conn.cursor()
        
        # Get keyword cache stats
        cursor.execute('SELECT COUNT(*) FROM keyword_cache')
        keyword_cache_count = cursor.fetchone()[0]
        
        # Get blog cache stats
        cursor.execute('SELECT COUNT(*) FROM blog_cache')
        blog_cache_count = cursor.fetchone()[0]
        
        # Get content calendar stats
        cursor.execute('SELECT COUNT(*) FROM content_calendar')
        calendar_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "cache_stats": {
                "keyword_queries": keyword_cache_count,
                "generated_blogs": blog_cache_count,
                "calendar_items": calendar_count
            },
            "api_status": {
                "google_suggest": "active",
                "openai_configured": OPENAI_API_KEY == os.getenv("HF_TOKEN"),
                "serp_api_configured": SERP_API_KEY == os.getenv("SERP_API_KEY")
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Complete workflow endpoint
@app.post("/api/complete-workflow")
async def complete_seo_workflow(seed_keyword: List[str], audience: str = "general", word_count: int = 1200):
    """Complete SEO blog generation workflow from keyword to published content"""
    try:
        workflow_results = {
            "seed_keyword": seed_keyword,
            "timestamp": datetime.now().isoformat(),
            "steps": {}
        }
        
        # Step 1: Get keywords
        keyword_request = KeywordRequest(seed_keywords=seed_keyword, limit=30)
        keywords_result = await get_keywords(keyword_request)
        workflow_results["steps"]["1_keywords"] = {
            "status": "completed",
            "keywords_found": len(keywords_result["keywords"]),
            "keywords": keywords_result["keywords"][:10]  # Show first 10
        }
        
        # Step 2: Generate topics
        topic_request = BlogTopicRequest(keywords=keywords_result["keywords"][:20], min_cluster_size=2)
        topics_result = await generate_blog_topics(topic_request)
        
        if not topics_result["blog_topics"]:
            logger.warning("No viable topics, falling back to seed keyword")
            best_topic = {
                "topic_title": f"Complete Guide to {seed_keyword[0].title()}",
                "main_keyword": seed_keyword[0],
                "related_keywords": keywords_result["keywords"][:10]
            }
            topics_result = {"blog_topics": [best_topic]}
        else:
            best_topic = topics_result["blog_topics"][0]
        
        # Step 3: Generate outline
        outline_request = BlogOutlineRequest(
            main_topic=best_topic["main_keyword"],
            target_keywords=best_topic["related_keywords"][:10],
            audience=audience
        )
        outline_result = await generate_blog_outline(outline_request)
        workflow_results["steps"]["3_outline"] = {
            "status": "completed",
            "sections": len(outline_result["outline"].get("main_sections", [])),
            "faqs": len(outline_result["outline"].get("faq", []))
        }
        
        # Step 4: Generate blog content
        blog_request = BlogGenerationRequest(
            topic=best_topic["topic_title"],
            outline=outline_result["outline"],
            target_keywords=best_topic["related_keywords"][:10],
            audience=audience,
            word_count=word_count
        )
        blog_result = await generate_blog_content(blog_request)
        workflow_results["steps"]["4_content"] = {
            "status": "completed",
            "word_count": blog_result.get("word_count", 0),
            "headings_count": len(blog_result.get("headings", []))
        }
        
        # Step 5: Generate content calendar suggestion
        calendar_request = ContentCalendarRequest(
            topics=[best_topic["topic_title"]],
            start_date=datetime.now().strftime("%Y-%m-%d"),
            frequency="weekly"
        )
        calendar_result = await generate_content_calendar(calendar_request)
        workflow_results["steps"]["5_calendar"] = {
            "status": "completed",
            "scheduled_date": calendar_result["calendar"][0]["scheduled_date"]
        }
        
        # Final results
        workflow_results["final_output"] = {
            "blog_title": blog_result["meta_title"],
            "meta_description": blog_result["meta_description"],
            "content": blog_result["content"],
            "word_count": blog_result.get("word_count", 0),
            "target_keywords": best_topic["related_keywords"][:5],
            "publishing_suggestion": calendar_result["calendar"][0]["scheduled_date"]
        }
        
        return workflow_results
        
    except Exception as e:
        logger.error(f"Complete workflow error: {e}")
        raise HTTPException(status_code=500, detail=f"Workflow error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Install required packages message
    required_packages = [
        "fastapi",
        "uvicorn",
        "sentence-transformers", 
        "scikit-learn",
        "numpy",
        "openai",
        "aiohttp",
        "beautifulsoup4",
        "sqlite3"  # Built-in with Python
    ]
    
    print("🚀 SEO Blog Generator API")
    print("=" * 50)
    print("\n📦 Required packages:")
    for package in required_packages:
        print(f"  - {package}")
    
    print(f"\n🛠️ Install with: pip install {' '.join(required_packages[:-1])}")
    
    print("\n🔑 API Keys needed:")
    print("  - OpenAI API Key (for blog generation)")
    print("  - SerpAPI Key (for 'People Also Ask' - optional)")
    
    print("\n🌐 Endpoints available:")
    print("  - POST /api/keywords - Get keyword suggestions") 
    print("  - POST /api/blog-topics - Cluster keywords into topics")
    print("  - POST /api/blog-outline - Generate SEO outline")
    print("  - POST /api/generate-blog - Create blog content")
    print("  - POST /api/complete-workflow - Full pipeline")
    print("  - GET /api/health - Health check")
    print("  - GET /api/stats - Usage statistics")
    
    print("\n🧪 Test the complete workflow:")
    print("  curl -X POST 'http://localhost:8000/api/complete-workflow?seed_keyword=best%20laptops&audience=students'")
    
    print("\n" + "=" * 50)
    print("Starting server on http://localhost:8000")
    print("API docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)