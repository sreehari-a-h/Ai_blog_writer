#!/usr/bin/env python3
"""
Enhanced SEO Blog Generator API - Dynamic content with styling and personalization
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
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced SEO Blog Generator API",
    description="Dynamic SEO blog content generation with styling and personalization",
    version="2.0.0"
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERP_API_KEY = os.getenv("SERP_API_KEY")

# Initialize models (lazy loading)
sentence_model = None
openai.api_key = OPENAI_API_KEY

def get_sentence_model():
    global sentence_model
    if sentence_model is None:
        sentence_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    return sentence_model

# Enhanced Pydantic models
class EnhancedKeywordRequest(BaseModel):
    seed_keywords: List[str]
    target_audience: str = "general"
    content_purpose: str = "informational"  # informational, commercial, transactional
    language: str = "en"
    country: str = "US"
    limit: int = 30

class ContentPersonalization(BaseModel):
    brand_name: Optional[str] = None
    company_website: Optional[str] = None
    social_media: Optional[Dict[str, str]] = None  # {"instagram": "handle", "twitter": "handle"}
    contact_info: Optional[Dict[str, str]] = None  # {"email": "", "phone": ""}
    author_bio: Optional[str] = None
    cta_preferences: Optional[List[str]] = None  # Custom CTAs

class EnhancedBlogRequest(BaseModel):
    topic: str
    primary_keywords: List[str]  # Main keywords to target
    secondary_keywords: List[str] = []  # Supporting keywords
    audience: str = "general"
    tone: str = "professional"  # professional, casual, authoritative, friendly, expert
    writing_style: str = "informative"  # informative, persuasive, storytelling, how-to, listicle
    word_count: int = 1200
    include_toc: bool = True
    include_faqs: bool = True
    include_conclusion_cta: bool = True
    personalization: Optional[ContentPersonalization] = None
    external_links_needed: Optional[List[str]] = None  # Topics that need external links
    
class DynamicOutlineRequest(BaseModel):
    topic: str
    primary_keywords: List[str]
    secondary_keywords: List[str] = []
    content_type: str = "guide"  # guide, comparison, review, how-to, listicle, news
    target_audience: str = "general"
    competitors_to_analyze: Optional[List[str]] = None  # URLs to analyze for structure

# Enhanced Database setup
def init_enhanced_db():
    conn = sqlite3.connect('enhanced_seo_cache.db')
    cursor = conn.cursor()
    
    # Enhanced keyword cache
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS enhanced_keyword_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            seed_keywords TEXT NOT NULL,
            target_audience TEXT,
            content_purpose TEXT,
            keywords_json TEXT NOT NULL,
            search_volume_data TEXT,
            competition_data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Blog templates and variations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS content_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_type TEXT NOT NULL,
            template_structure TEXT NOT NULL,
            style_guide TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # User personalization data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_personalization (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            brand_info TEXT,
            style_preferences TEXT,
            content_history TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

init_enhanced_db()

class EnhancedAIBlogWriter:
    """Enhanced AI blog writer with dynamic content generation and styling"""

    def __init__(self):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",    
            api_key=os.getenv("HF_TOKEN")
        )
        
        # Content variation templates
        self.heading_styles = {
            "question": ["How to", "What is", "Why", "When should", "Where to"],
            "benefit": ["Benefits of", "Advantages of", "Why you need", "Power of"],
            "guide": ["Complete Guide to", "Ultimate", "Definitive", "Comprehensive"],
            "comparison": ["vs", "compared", "Best", "Top", "versus"],
            "problem_solution": ["Common Problems", "Solutions for", "Fixing", "Troubleshooting"]
        }
        
        self.content_enhancers = {
            "emojis": {
                "tech": ["💻", "⚡", "🚀", "🔧", "💡", "📱", "🌐", "🔥"],
                "business": ["💼", "📈", "💰", "🎯", "📊", "🏆", "💎", "🚀"],
                "health": ["💪", "🍎", "🌿", "❤️", "🧘", "🌟", "✨", "🏃"],
                "education": ["📚", "🎓", "💡", "📝", "🧠", "🔍", "📖", "⭐"],
                "lifestyle": ["🌟", "✨", "🎉", "🎨", "🌈", "💫", "🦋", "🌸"]
            },
            "power_words": {
                "action": ["Transform", "Master", "Unlock", "Discover", "Achieve", "Dominate"],
                "benefit": ["Essential", "Proven", "Effective", "Powerful", "Ultimate", "Complete"],
                "urgency": ["Now", "Today", "Instantly", "Immediately", "Quickly", "Fast"]
            }
        }

    async def generate_dynamic_blog(self, request: EnhancedBlogRequest) -> Dict[str, Any]:
        """Generate dynamic blog content with styling and personalization"""
        
        try:
            # Step 1: Create dynamic outline based on content type
            outline = await self._create_dynamic_outline(request)
            
            # Step 2: Generate content with styling
            content = await self._generate_styled_content(request, outline)
            
            # Step 3: Add personalization elements
            personalized_content = self._add_personalization(content, request.personalization)
            
            # Step 4: Format with rich styling
            styled_content = self._apply_rich_styling(personalized_content, request)
            
            # Step 5: Generate metadata
            metadata = await self._generate_seo_metadata(request, outline)
            
            return {
                "content": styled_content,
                "metadata": metadata,
                "outline_used": outline,
                "styling_applied": True,
                "personalization_level": "high" if request.personalization else "standard",
                "external_links_needed": self._identify_external_links(styled_content),
                "social_media_integration": self._get_social_integration_points(styled_content),
                "word_count": len(styled_content.split()),
                "generated_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Enhanced blog generation error: {e}")
            return await self._generate_fallback_content(request)

    async def _create_dynamic_outline(self, request: EnhancedBlogRequest) -> Dict[str, Any]:
        """Create dynamic, AI-powered outline structure"""
        
        content_type = request.writing_style
        primary_kw = request.primary_keywords[0] if request.primary_keywords else request.topic
        
        # Let AI decide the structure based on topic and keywords
        outline_prompt = f"""
Create a comprehensive, SEO-optimized blog outline for:

TOPIC: {request.topic}
PRIMARY KEYWORDS: {', '.join(request.primary_keywords)}
SECONDARY KEYWORDS: {', '.join(request.secondary_keywords)}
CONTENT TYPE: {content_type}
TARGET AUDIENCE: {request.audience}
TONE: {request.tone}

Requirements:
1. Create engaging H1 that includes primary keyword naturally
2. Generate 4-6 H2 sections that flow logically and target different keywords
3. Include H3 subsections where appropriate
4. Suggest where to use bold, italics, and emojis naturally
5. Plan FAQ section with 5-7 relevant questions
6. Design compelling introduction hook and conclusion CTA
7. Identify spots for external links, statistics, or expert quotes

Return a JSON structure with:
- h1: Main heading
- introduction: Hook and preview
- main_sections: Array of sections with h2, h3, styling_notes, keywords_to_target
- faq_section: Questions and answer previews  
- conclusion: Summary and CTA
- styling_recommendations: Where to use emojis, bold, italics
- external_link_opportunities: Suggested external references needed
"""

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="Qwen/Qwen3-Next-80B-A3B-Instruct:novita",
                messages=[
                    {"role": "system", "content": "You are an expert SEO content strategist and outline creator. Return only valid JSON."},
                    {"role": "user", "content": outline_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
            )
            
            outline_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON, fallback to structured outline if needed
            try:
                outline = json.loads(outline_text)
            except json.JSONDecodeError:
                outline = self._parse_outline_fallback(outline_text, request)
                
            return outline
            
        except Exception as e:
            logger.error(f"Dynamic outline generation failed: {e}")
            return self._create_fallback_outline(request)

    async def _generate_styled_content(self, request: EnhancedBlogRequest, outline: Dict[str, Any]) -> str:
        """Generate full blog content with rich styling"""
        
        # Enhanced content generation prompt
        content_prompt = f"""
Write a comprehensive, engaging blog post based on this outline:

OUTLINE: {json.dumps(outline, indent=2)}

REQUIREMENTS:
1. **Rich Formatting**: Use Markdown with bold, italics, emojis appropriately
2. **Keyword Integration**: Naturally include these keywords:
   - Primary: {', '.join(request.primary_keywords)}
   - Secondary: {', '.join(request.secondary_keywords)}
3. **Styling Guidelines**:
   - Use emojis to enhance headings and key points (but don't overuse)
   - Bold important terms and key takeaways
   - Italicize emphasis and quotes
   - Use bullet points and numbered lists effectively
   - Add call-out boxes with important tips
4. **Content Quality**:
   - Write {request.word_count} words approximately
   - {request.tone} tone for {request.audience} audience
   - Include actionable insights and practical examples
   - Vary sentence structure and paragraph length
5. **SEO Optimization**:
   - Natural keyword placement (avoid stuffing)
   - Use semantic keywords and related terms
   - Include internal linking suggestions [LINK: anchor text]
   - Add spots for external links [EXTERNAL: topic]

OUTPUT FORMAT:
Return the complete blog post in Markdown format with proper styling.
Include Table of Contents if requested: {request.include_toc}
Include FAQ section if requested: {request.include_faqs}
"""

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="Qwen/Qwen3-Next-80B-A3B-Instruct:novita",
                messages=[
                    {"role": "system", "content": f"You are an expert {request.tone} blog writer specializing in {request.audience} content. Write engaging, well-styled content with proper Markdown formatting."},
                    {"role": "user", "content": content_prompt}
                ],
                temperature=0.8,
                max_tokens=min(6000, request.word_count * 3),
            )
            
            content = response.choices[0].message.content.strip()
            return content
            
        except Exception as e:
            logger.error(f"Styled content generation failed: {e}")
            return self._generate_fallback_styled_content(request, outline)

    def _add_personalization(self, content: str, personalization: Optional[ContentPersonalization]) -> str:
        """Add brand and personal elements to content"""
        
        if not personalization:
            return content
            
        personalized_content = content
        
        # Add brand mentions
        if personalization.brand_name:
            # Add brand references naturally
            brand_mentions = [
                f"At **{personalization.brand_name}**, we believe",
                f"Our team at {personalization.brand_name} has found",
                f"{personalization.brand_name}'s experience shows"
            ]
            
            # Insert brand mention in appropriate location
            paragraphs = content.split('\n\n')
            if len(paragraphs) > 2:
                insert_pos = len(paragraphs) // 3
                brand_mention = random.choice(brand_mentions)
                paragraphs[insert_pos] = f"{brand_mention} that {paragraphs[insert_pos].lower()}"
                personalized_content = '\n\n'.join(paragraphs)
        
        # Add author bio section
        if personalization.author_bio:
            author_section = f"""
---

## About the Author

{personalization.author_bio}
"""
            personalized_content += author_section
        
        # Add social media integration points
        if personalization.social_media:
            social_cta = "\n\n**Stay Connected:**\n"
            for platform, handle in personalization.social_media.items():
                emoji = {"instagram": "📱", "twitter": "🐦", "linkedin": "💼", "facebook": "👥"}.get(platform, "🌐")
                social_cta += f"- {emoji} Follow us on {platform.title()}: @{handle}\n"
            
            # Insert before conclusion
            if "## Conclusion" in personalized_content:
                personalized_content = personalized_content.replace("## Conclusion", social_cta + "\n## Conclusion")
            else:
                personalized_content += social_cta
        
        return personalized_content

    def _apply_rich_styling(self, content: str, request: EnhancedBlogRequest) -> str:
        """Apply advanced styling and formatting"""
        
        styled_content = content
        
        # Enhance headings with emojis if appropriate
        topic_category = self._categorize_topic(request.topic)
        emoji_set = self.content_enhancers["emojis"].get(topic_category, ["✨", "🌟", "💡"])
        
        # Add strategic emojis to H2 headings
        h2_pattern = r'^(## )(.+)$'
        def add_emoji_to_h2(match):
            emoji = random.choice(emoji_set)
            return f"{match.group(1)}{emoji} {match.group(2)}"
        
        styled_content = re.sub(h2_pattern, add_emoji_to_h2, styled_content, flags=re.MULTILINE, count=3)
        
        # Enhance key points with styling
        styled_content = self._enhance_key_points(styled_content)
        
        # Add call-out boxes
        styled_content = self._add_callout_boxes(styled_content)
        
        # Format lists better
        styled_content = self._enhance_lists(styled_content)
        
        return styled_content

    def _categorize_topic(self, topic: str) -> str:
        """Categorize topic to select appropriate emojis"""
        topic_lower = topic.lower()
        
        tech_keywords = ["software", "app", "digital", "tech", "computer", "online", "web"]
        business_keywords = ["business", "marketing", "sales", "entrepreneur", "money", "finance"]
        health_keywords = ["health", "fitness", "wellness", "diet", "exercise", "mental"]
        education_keywords = ["learn", "education", "study", "course", "skill", "training"]
        
        if any(keyword in topic_lower for keyword in tech_keywords):
            return "tech"
        elif any(keyword in topic_lower for keyword in business_keywords):
            return "business"
        elif any(keyword in topic_lower for keyword in health_keywords):
            return "health"
        elif any(keyword in topic_lower for keyword in education_keywords):
            return "education"
        else:
            return "lifestyle"

    def _enhance_key_points(self, content: str) -> str:
        """Enhance important points with better styling"""
        
        # Bold important phrases
        power_patterns = [
            (r'\b(important|crucial|essential|key|vital|critical)\b', r'**\1**'),
            (r'\b(best|top|ultimate|perfect|ideal)\b', r'**\1**'),
            (r'\b(remember|note|warning|tip)\b', r'**⚠️ \1**'),
        ]
        
        enhanced_content = content
        for pattern, replacement in power_patterns:
            enhanced_content = re.sub(pattern, replacement, enhanced_content, flags=re.IGNORECASE)
        
        return enhanced_content

    def _add_callout_boxes(self, content: str) -> str:
        """Add callout boxes for important information"""
        
        # Look for tip/note patterns and enhance them
        tip_pattern = r'^(Tip|Note|Warning|Important):\s*(.+)$'
        
        def create_callout(match):
            tip_type = match.group(1)
            tip_content = match.group(2)
            
            emoji_map = {
                "Tip": "💡",
                "Note": "📝", 
                "Warning": "⚠️",
                "Important": "❗"
            }
            
            emoji = emoji_map.get(tip_type, "ℹ️")
            
            return f"""
> {emoji} **{tip_type}:**
> {tip_content}
"""
        
        enhanced_content = re.sub(tip_pattern, create_callout, content, flags=re.MULTILINE)
        
        return enhanced_content

    def _enhance_lists(self, content: str) -> str:
        """Make lists more visually appealing"""
        
        # Enhance bullet points with emojis occasionally
        bullet_pattern = r'^- (.+)$'
        
        def maybe_add_bullet_emoji(match):
            # Add emoji to every 3rd bullet point
            if random.random() < 0.3:  # 30% chance
                return f"- ✅ {match.group(1)}"
            return match.group(0)
        
        enhanced_content = re.sub(bullet_pattern, maybe_add_bullet_emoji, content, flags=re.MULTILINE)
        
        return enhanced_content

    async def _generate_seo_metadata(self, request: EnhancedBlogRequest, outline: Dict[str, Any]) -> Dict[str, str]:
        """Generate comprehensive SEO metadata"""
        
        primary_kw = request.primary_keywords[0] if request.primary_keywords else request.topic
        
        # AI-generated meta title and description
        meta_prompt = f"""
Create SEO metadata for this blog post:

Topic: {request.topic}
Primary Keyword: {primary_kw}
Content Type: {request.writing_style}
Target Audience: {request.audience}

Generate:
1. Meta Title (max 60 characters, include primary keyword)
2. Meta Description (max 160 characters, compelling and keyword-rich)
3. Focus Keyphrase: Main keyword phrase to target
4. SEO Title (for social sharing, can be longer)
5. Social Description (for social media sharing)

Make them compelling and click-worthy while being accurate to the content.
Return as JSON with keys: meta_title, meta_description, focus_keyphrase, seo_title, social_description
"""

        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="Qwen/Qwen3-Next-80B-A3B-Instruct:novita",
                messages=[
                    {"role": "system", "content": "You are an SEO metadata expert. Return only valid JSON."},
                    {"role": "user", "content": meta_prompt}
                ],
                temperature=0.7,
                max_tokens=500,
            )
            
            metadata_text = response.choices[0].message.content.strip()
            
            try:
                metadata = json.loads(metadata_text)
            except json.JSONDecodeError:
                metadata = self._create_fallback_metadata(primary_kw, request.topic)
                
            return metadata
            
        except Exception as e:
            logger.error(f"SEO metadata generation failed: {e}")
            return self._create_fallback_metadata(primary_kw, request.topic)

    def _identify_external_links(self, content: str) -> List[Dict[str, str]]:
        """Identify where external links are needed"""
        
        external_link_pattern = r'\[EXTERNAL:\s*([^\]]+)\]'
        matches = re.findall(external_link_pattern, content)
        
        external_links = []
        for match in matches:
            external_links.append({
                "topic": match.strip(),
                "suggestion": f"Find authoritative source about {match.strip()}",
                "anchor_text": match.strip()
            })
        
        return external_links

    def _get_social_integration_points(self, content: str) -> List[Dict[str, str]]:
        """Identify points where social media integration would be effective"""
        
        integration_points = []
        
        # Look for sections that would benefit from social proof
        social_triggers = [
            "share your experience",
            "what do you think",
            "have you tried",
            "tell us about",
            "join the conversation"
        ]
        
        for trigger in social_triggers:
            if trigger in content.lower():
                integration_points.append({
                    "type": "engagement",
                    "location": f"Near: '{trigger}'",
                    "suggestion": "Add social media engagement prompt"
                })
        
        # Add strategic sharing points
        if "## Conclusion" in content:
            integration_points.append({
                "type": "sharing",
                "location": "Before conclusion",
                "suggestion": "Add social sharing buttons and encouragement"
            })
        
        return integration_points

    # Fallback methods
    def _create_fallback_outline(self, request: EnhancedBlogRequest) -> Dict[str, Any]:
        """Create fallback outline when AI generation fails"""
        
        primary_kw = request.primary_keywords[0] if request.primary_keywords else request.topic
        
        return {
            "h1": f"Complete Guide to {primary_kw.title()}",
            "introduction": {
                "hook": f"Everything you need to know about {primary_kw}",
                "preview": "Comprehensive guide with actionable insights"
            },
            "main_sections": [
                {
                    "h2": f"What is {primary_kw.title()}?",
                    "h3": ["Definition", "Key concepts", "Why it matters"],
                    "keywords_to_target": request.primary_keywords[:2]
                },
                {
                    "h2": f"Benefits of {primary_kw.title()}",
                    "h3": ["Main advantages", "Impact on users", "Long-term benefits"],
                    "keywords_to_target": [kw for kw in request.secondary_keywords if "benefit" in kw.lower()][:2]
                },
                {
                    "h2": f"How to Get Started with {primary_kw.title()}",
                    "h3": ["Step-by-step guide", "Best practices", "Common mistakes"],
                    "keywords_to_target": [kw for kw in request.secondary_keywords if "how to" in kw.lower()][:2]
                }
            ],
            "faq_section": [
                f"What is {primary_kw}?",
                f"How does {primary_kw} work?",
                f"Is {primary_kw} worth it?",
                f"How much does {primary_kw} cost?",
                f"What are the alternatives to {primary_kw}?"
            ],
            "conclusion": {
                "summary": f"Key takeaways about {primary_kw}",
                "cta": "Ready to get started?"
            }
        }

    def _create_fallback_metadata(self, primary_kw: str, topic: str) -> Dict[str, str]:
        """Create fallback SEO metadata"""
        
        return {
            "meta_title": f"{primary_kw.title()} - Complete Guide {datetime.now().year}",
            "meta_description": f"Learn everything about {primary_kw}. Comprehensive guide with tips, best practices, and expert insights. Get started today!",
            "focus_keyphrase": primary_kw,
            "seo_title": f"Ultimate {primary_kw.title()} Guide - Everything You Need to Know",
            "social_description": f"Discover the complete guide to {primary_kw}. Expert tips, best practices, and actionable insights included!"
        }

    async def _generate_fallback_content(self, request: EnhancedBlogRequest) -> Dict[str, Any]:
        """Generate fallback content when main generation fails"""
        
        primary_kw = request.primary_keywords[0] if request.primary_keywords else request.topic
        
        fallback_content = f"""# 🚀 Complete Guide to {primary_kw.title()}

## Introduction

Welcome to the most comprehensive guide on **{primary_kw}**! Whether you're a beginner or looking to advance your knowledge, this guide covers everything you need to know.

## 💡 What is {primary_kw.title()}?

**{primary_kw.title()}** is {self._get_generic_definition(primary_kw)}. Understanding this concept is crucial for anyone looking to succeed in this area.

### Key Benefits ✨

- **Improved efficiency** and better results
- **Enhanced understanding** of core principles  
- **Practical applications** you can use immediately
- **Long-term value** for your goals

## 🎯 Getting Started

Here's how to begin your journey with **{primary_kw}**:

1. **Research and Planning** - Understand your specific needs
2. **Choose Your Approach** - Select the right strategy
3. **Implementation** - Take action with confidence
4. **Monitor Progress** - Track and optimize your results

> 💡 **Pro Tip:** Start small and gradually expand your efforts for the best results.

## ❓ Frequently Asked Questions

### What makes {primary_kw} effective?
The effectiveness comes from its proven methodology and widespread adoption across industries.

### How long does it take to see results?
Results typically vary based on implementation, but most users see improvements within the first few weeks.

### Is {primary_kw} suitable for beginners?
Absolutely! This guide is designed to help users at all levels, from complete beginners to advanced practitioners.

## 🎉 Conclusion

**{primary_kw.title()}** offers tremendous opportunities for growth and success. By following the strategies outlined in this guide, you'll be well-equipped to achieve your goals.

**Ready to get started?** Take action today and begin implementing these proven strategies!

---

*Need help with implementation? Share your questions in the comments below!*
"""

        return {
            "content": fallback_content,
            "metadata": self._create_fallback_metadata(primary_kw, request.topic),
            "outline_used": self._create_fallback_outline(request),
            "styling_applied": True,
            "personalization_level": "standard",
            "external_links_needed": [],
            "social_media_integration": [],
            "word_count": len(fallback_content.split()),
            "generated_at": datetime.now().isoformat()
        }

    def _get_generic_definition(self, keyword: str) -> str:
        """Generate generic definition for keywords"""
        definitions = [
            f"a comprehensive approach that delivers measurable results",
            f"an essential strategy used by professionals and experts",
            f"a proven methodology that has helped thousands achieve success",
            f"an innovative solution that addresses common challenges",
            f"a systematic approach that maximizes efficiency and effectiveness"
        ]
        return random.choice(definitions)

# Enhanced API Endpoints

@app.post("/api/enhanced-blog")
async def generate_enhanced_blog(request: EnhancedBlogRequest):
    """Generate enhanced blog with dynamic styling and personalization"""
    try:
        blog_writer = EnhancedAIBlogWriter()
        result = await blog_writer.generate_dynamic_blog(request)
        
        return {
            "success": True,
            "blog_data": result,
            "user_actions_needed": {
                "external_links": result["external_links_needed"],
                "social_media_setup": result["social_media_integration"],
                "personalization_opportunities": _get_personalization_suggestions(request)
            },
            "content_analysis": {
                "readability_score": _calculate_readability_score(result["content"]),
                "seo_score": _calculate_seo_score(result["content"], request.primary_keywords),
                "styling_completeness": _analyze_styling(result["content"])
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced blog generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Blog generation failed: {str(e)}")

@app.post("/api/request-external-links")
async def request_external_links(
    external_links: List[Dict[str, str]],
    user_provided_links: Optional[Dict[str, str]] = None
):
    """Handle external links that users need to provide"""
    try:
        processed_links = []
        missing_links = []
        
        for link_request in external_links:
            topic = link_request.get("topic", "")
            
            if user_provided_links and topic in user_provided_links:
                processed_links.append({
                    "topic": topic,
                    "url": user_provided_links[topic],
                    "status": "provided",
                    "anchor_text": link_request.get("anchor_text", topic)
                })
            else:
                missing_links.append({
                    "topic": topic,
                    "description": f"Please provide an authoritative link about {topic}",
                    "suggested_anchor": link_request.get("anchor_text", topic),
                    "type": "external_reference"
                })
        
        return {
            "processed_links": processed_links,
            "missing_links": missing_links,
            "next_steps": "Please provide URLs for the missing external references to complete your blog post."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"External links processing failed: {str(e)}")

@app.post("/api/request-social-media")
async def request_social_media_info(
    social_integration_points: List[Dict[str, str]],
    user_social_media: Optional[Dict[str, str]] = None
):
    """Handle social media integration requests"""
    try:
        integrated_points = []
        missing_social_info = []
        
        for integration_point in social_integration_points:
            point_type = integration_point.get("type", "")
            
            if user_social_media:
                social_snippet = _generate_social_snippet(point_type, user_social_media)
                integrated_points.append({
                    "location": integration_point.get("location", ""),
                    "type": point_type,
                    "social_snippet": social_snippet,
                    "status": "ready"
                })
            else:
                missing_social_info.append({
                    "type": point_type,
                    "location": integration_point.get("location", ""),
                    "needed_info": _get_social_requirements(point_type)
                })
        
        return {
            "integrated_points": integrated_points,
            "missing_social_info": missing_social_info,
            "social_platforms_supported": ["instagram", "twitter", "linkedin", "facebook", "youtube", "tiktok", "whatsapp"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Social media integration failed: {str(e)}")

@app.post("/api/finalize-blog")
async def finalize_blog_with_user_inputs(
    blog_content: str,
    external_links: Optional[Dict[str, str]] = None,
    social_media: Optional[Dict[str, str]] = None,
    additional_styling: Optional[Dict[str, Any]] = None
):
    """Finalize blog content with user-provided external links and social media"""
    try:
        finalized_content = blog_content
        
        # Insert external links
        if external_links:
            for topic, url in external_links.items():
                # Replace [EXTERNAL: topic] placeholders with actual links
                external_pattern = f"\\[EXTERNAL:\\s*{re.escape(topic)}\\]"
                link_replacement = f"[{topic}]({url})"
                finalized_content = re.sub(external_pattern, link_replacement, finalized_content, flags=re.IGNORECASE)
        
        # Add social media integration
        if social_media:
            social_section = _create_social_media_section(social_media)
            
            # Insert before conclusion or at the end
            if "## Conclusion" in finalized_content:
                finalized_content = finalized_content.replace("## Conclusion", f"{social_section}\n\n## Conclusion")
            else:
                finalized_content += f"\n\n{social_section}"
        
        # Apply additional styling if requested
        if additional_styling:
            finalized_content = _apply_additional_styling(finalized_content, additional_styling)
        
        # Generate final copy-ready format
        copy_ready_content = _format_for_copying(finalized_content)
        
        return {
            "finalized_content": finalized_content,
            "copy_ready_content": copy_ready_content,
            "statistics": {
                "word_count": len(finalized_content.split()),
                "external_links_added": len(external_links) if external_links else 0,
                "social_platforms_integrated": len(social_media) if social_media else 0,
                "headings_count": len(re.findall(r'^#+\s', finalized_content, re.MULTILINE))
            },
            "finalization_complete": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blog finalization failed: {str(e)}")

# Helper functions
def _get_personalization_suggestions(request: EnhancedBlogRequest) -> List[Dict[str, str]]:
    """Generate personalization suggestions based on content"""
    suggestions = []
    
    if not request.personalization or not request.personalization.brand_name:
        suggestions.append({
            "type": "branding",
            "suggestion": "Add your brand name for better recognition and authority",
            "benefit": "Increases brand awareness and trust"
        })
    
    if not request.personalization or not request.personalization.social_media:
        suggestions.append({
            "type": "social_media",
            "suggestion": "Provide social media handles to increase engagement",
            "benefit": "Builds community and drives traffic to your social profiles"
        })
    
    if not request.personalization or not request.personalization.author_bio:
        suggestions.append({
            "type": "author_bio",
            "suggestion": "Include author bio to establish expertise and credibility",
            "benefit": "Builds trust and establishes thought leadership"
        })
    
    return suggestions

def _calculate_readability_score(content: str) -> Dict[str, Any]:
    """Calculate basic readability metrics"""
    sentences = content.count('.') + content.count('!') + content.count('?')
    words = len(content.split())
    paragraphs = len([p for p in content.split('\n\n') if p.strip()])
    
    avg_words_per_sentence = words / max(sentences, 1)
    avg_sentences_per_paragraph = sentences / max(paragraphs, 1)
    
    # Simple readability assessment
    if avg_words_per_sentence < 15:
        readability_level = "Easy"
    elif avg_words_per_sentence < 25:
        readability_level = "Medium"
    else:
        readability_level = "Complex"
    
    return {
        "readability_level": readability_level,
        "avg_words_per_sentence": round(avg_words_per_sentence, 1),
        "avg_sentences_per_paragraph": round(avg_sentences_per_paragraph, 1),
        "total_words": words,
        "total_paragraphs": paragraphs
    }

def _calculate_seo_score(content: str, primary_keywords: List[str]) -> Dict[str, Any]:
    """Calculate basic SEO score"""
    content_lower = content.lower()
    
    keyword_density = {}
    total_words = len(content.split())
    
    for keyword in primary_keywords:
        keyword_count = content_lower.count(keyword.lower())
        density = (keyword_count / total_words) * 100 if total_words > 0 else 0
        keyword_density[keyword] = {
            "count": keyword_count,
            "density": round(density, 2)
        }
    
    # SEO factors
    has_h1 = bool(re.search(r'^#\s', content, re.MULTILINE))
    has_h2 = bool(re.search(r'^##\s', content, re.MULTILINE))
    has_internal_links = "[LINK:" in content
    has_external_links = "[EXTERNAL:" in content or "](http" in content
    
    seo_score = 0
    if has_h1: seo_score += 20
    if has_h2: seo_score += 15
    if has_internal_links: seo_score += 10
    if has_external_links: seo_score += 10
    if any(kd["density"] > 0.5 for kd in keyword_density.values()): seo_score += 15
    if total_words > 800: seo_score += 10
    if "FAQ" in content or "frequently asked" in content.lower(): seo_score += 10
    if len(re.findall(r'^#+\s', content, re.MULTILINE)) >= 4: seo_score += 10
    
    return {
        "overall_score": min(seo_score, 100),
        "keyword_density": keyword_density,
        "seo_factors": {
            "has_h1_heading": has_h1,
            "has_h2_headings": has_h2,
            "has_internal_links": has_internal_links,
            "has_external_links": has_external_links,
            "sufficient_length": total_words > 800,
            "has_faq_section": "FAQ" in content or "frequently asked" in content.lower()
        }
    }

def _analyze_styling(content: str) -> Dict[str, Any]:
    """Analyze styling completeness"""
    
    styling_elements = {
        "bold_text": bool(re.search(r'\*\*[^*]+\*\*', content)),
        "italic_text": bool(re.search(r'\*[^*]+\*', content)) and not bool(re.search(r'\*\*[^*]+\*\*', content)),
        "bullet_points": bool(re.search(r'^\s*[-*]\s', content, re.MULTILINE)),
        "numbered_lists": bool(re.search(r'^\s*\d+\.\s', content, re.MULTILINE)),
        "headings": bool(re.search(r'^#+\s', content, re.MULTILINE)),
        "callout_boxes": bool(re.search(r'^>\s', content, re.MULTILINE)),
        "code_blocks": bool(re.search(r'```', content)),
        "links": bool(re.search(r'\[([^\]]+)\]\([^)]+\)', content))
    }
    
    styling_score = sum(styling_elements.values()) * 12.5  # Max 100
    
    return {
        "styling_score": styling_score,
        "elements_used": styling_elements,
        "suggestions": _get_styling_suggestions(styling_elements)
    }

def _get_styling_suggestions(styling_elements: Dict[str, bool]) -> List[str]:
    """Get suggestions for improving styling"""
    suggestions = []
    
    if not styling_elements["bold_text"]:
        suggestions.append("Add bold text for emphasis on key points")
    
    if not styling_elements["bullet_points"] and not styling_elements["numbered_lists"]:
        suggestions.append("Use bullet points or numbered lists for better readability")
    
    if not styling_elements["callout_boxes"]:
        suggestions.append("Add callout boxes for important tips or notes")
    
    if not styling_elements["links"]:
        suggestions.append("Include relevant internal and external links")
    
    return suggestions

def _generate_social_snippet(point_type: str, social_media: Dict[str, str]) -> str:
    """Generate social media snippet based on type"""
    
    if point_type == "engagement":
        platforms = []
        for platform, handle in social_media.items():
            platform_emoji = {"instagram": "📷", "twitter": "🐦", "linkedin": "💼", "facebook": "👥"}.get(platform, "🌐")
            platforms.append(f"{platform_emoji} @{handle}")
        
        return f"💬 **Join the conversation!** Share your thoughts with us on {', '.join(platforms)}"
    
    elif point_type == "sharing":
        return "🚀 **Found this helpful?** Share it with others who might benefit from these insights!"
    
    return "🌟 **Stay connected** with us for more valuable content and updates!"

def _get_social_requirements(point_type: str) -> List[str]:
    """Get requirements for social media integration"""
    
    if point_type == "engagement":
        return [
            "Social media handles (Instagram, Twitter, LinkedIn, etc.)",
            "Preferred engagement message",
            "Platform-specific hashtags (optional)"
        ]
    
    elif point_type == "sharing":
        return [
            "Preferred sharing message",
            "Social media handles for tagging",
            "Branded hashtags (optional)"
        ]
    
    return [
        "Social media handles",
        "Contact preferences"
    ]

def _create_social_media_section(social_media: Dict[str, str]) -> str:
    """Create a social media section"""
    
    social_section = "\n---\n\n## 🌟 Stay Connected\n\n"
    
    platform_emojis = {
        "instagram": "📸",
        "twitter": "🐦", 
        "linkedin": "💼",
        "facebook": "👥",
        "youtube": "📺",
        "tiktok": "🎵",
        "whatsapp": "💬"
    }
    
    for platform, handle in social_media.items():
        emoji = platform_emojis.get(platform.lower(), "🌐")
        platform_name = platform.title()
        
        if platform.lower() == "whatsapp":
            social_section += f"- {emoji} **{platform_name}**: [Contact us]({handle})\n"
        else:
            social_section += f"- {emoji} **{platform_name}**: [@{handle}](https://{platform.lower()}.com/{handle})\n"
    
    social_section += "\n*Follow us for more insights, tips, and updates!*\n"
    
    return social_section

def _apply_additional_styling(content: str, styling_options: Dict[str, Any]) -> str:
    """Apply additional styling based on user preferences"""
    
    styled_content = content
    
    # Add more emphasis if requested
    if styling_options.get("extra_emphasis", False):
        # Make first sentence of each paragraph bold
        paragraphs = styled_content.split('\n\n')
        for i, paragraph in enumerate(paragraphs):
            if paragraph.strip() and not paragraph.startswith('#'):
                sentences = paragraph.split('. ')
                if sentences:
                    sentences[0] = f"**{sentences[0]}**"
                    paragraphs[i] = '. '.join(sentences)
        styled_content = '\n\n'.join(paragraphs)
    
    # Add reading time estimate
    if styling_options.get("reading_time", False):
        word_count = len(styled_content.split())
        reading_time = max(1, round(word_count / 200))  # Average 200 words per minute
        
        reading_time_note = f"\n> ⏱️ **Reading time**: Approximately {reading_time} minute{'s' if reading_time > 1 else ''}\n\n"
        
        # Insert after first heading
        first_heading_match = re.search(r'^#[^#].*', styled_content, re.MULTILINE)
        if first_heading_match:
            insert_pos = first_heading_match.end()
            styled_content = styled_content[:insert_pos] + reading_time_note + styled_content[insert_pos:]
    
    return styled_content

def _format_for_copying(content: str) -> str:
    """Format content for easy copying with preserved styling"""
    
    # Ensure proper line spacing
    formatted_content = re.sub(r'\n{3,}', '\n\n', content)
    
    # Add copy instructions at the top
    copy_instructions = """<!-- 
COPY-READY BLOG POST
- All styling is preserved in Markdown format
- Bold text: **text**
- Italic text: *text*
- Links: [text](url)
- Headings: # H1, ## H2, ### H3
- Lists: - bullet or 1. numbered
- Callouts: > quoted text
-->

"""
    
    return copy_instructions + formatted_content

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Enhanced SEO Blog Generator API v2.0")
    print("=" * 60)
    print("\n✨ NEW FEATURES:")
    print("  - Dynamic AI-powered content structure")
    print("  - Rich styling with emojis, bold, italics")
    print("  - Personalized brand integration")
    print("  - External links management")
    print("  - Social media integration")
    print("  - Copy-ready formatted output")
    print("  - Advanced SEO analysis")
    
    print("\n📚 Key Endpoints:")
    print("  - POST /api/enhanced-blog - Generate dynamic styled blog")
    print("  - POST /api/request-external-links - Handle external references")  
    print("  - POST /api/request-social-media - Social media integration")
    print("  - POST /api/finalize-blog - Finalize with user inputs")
    
    print("\n🎯 Example Enhanced Request:")
    print("""
{
  "topic": "Best Project Management Tools",
  "primary_keywords": ["project management tools", "best PM software"],
  "secondary_keywords": ["team collaboration", "productivity apps"],
  "audience": "small business owners", 
  "tone": "friendly",
  "writing_style": "informative",
  "word_count": 1500,
  "include_toc": true,
  "include_faqs": true,
  "personalization": {
    "brand_name": "ProductivityPro",
    "social_media": {
      "instagram": "productivitypro",
      "linkedin": "productivitypro-official"
    },
    "author_bio": "Tech expert with 10 years in PM solutions"
  }
}
""")
    
    print("\n" + "=" * 60)
    print("🌟 Starting enhanced server on http://localhost:8000")
    print("📖 Interactive docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)