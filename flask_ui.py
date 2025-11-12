#!/usr/bin/env python3
"""
SEO Blog Generator - Flask Web UI
MVP interface to interact with FastAPI backend
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import requests
import json
import markdown
import re
from datetime import datetime
import os
from markupsafe import Markup

app = Flask(__name__)
app.secret_key = 'seo-blog-generator-secret-key-2024'

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"  # Your FastAPI server
UPLOAD_FOLDER = 'generated_blogs'

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Sample data for quick testing
SAMPLE_DATA = {
    "dubai_real_estate": {
        "topic": "Best Dubai Real Estate Agents for Off-Plan Properties",
        "primary_keywords": ["Dubai real estate agents", "off-plan properties Dubai", "RERA licensed agents"],
        "secondary_keywords": ["Dubai property investment", "real estate brokers Dubai", "property consultants Dubai", "Dubai Marina agents", "Downtown Dubai properties", "Dubai Hills Estate", "investment properties UAE"],
        "audience": "property investors and buyers",
        "tone": "professional",
        "writing_style": "informative",
        "word_count": 1500,
        "personalization": {
            "brand_name": "OffPlan Market",
            "company_website": "https://offplan.market",
            "social_media": {
                "instagram": "offplanmarket",
                "linkedin": "offplan-market-dubai", 
                "twitter": "offplanmarket"
            },
            "contact_info": {
                "email": "info@offplan.market",
                "phone": "+971-4-123-4567",
                "whatsapp": "+971-50-123-4567"
            },
            "author_bio": "Property investment specialist with 8+ years experience in Dubai's off-plan market.",
            "cta_preferences": ["Explore our curated off-plan properties", "Schedule a consultation with our experts"]
        }
    },
    "tech_saas": {
        "topic": "Best Project Management Software for Remote Teams",
        "primary_keywords": ["project management software", "remote team tools", "collaboration apps"],
        "secondary_keywords": ["productivity software", "team management", "workflow automation", "task tracking", "remote work tools"],
        "audience": "startup founders and team leads",
        "tone": "friendly",
        "writing_style": "informative",
        "word_count": 1200,
        "personalization": {
            "brand_name": "ProductivityPro",
            "social_media": {
                "twitter": "productivitypro",
                "linkedin": "productivitypro-official"
            }
        }
    }
}

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html', sample_data=SAMPLE_DATA)

seo_score = 0  # Global variable to hold SEO score for preview

@app.route('/generate', methods=['POST'])
def generate_blog():
    """Generate blog via FastAPI"""
    try:
        # Get form data
        form_data = request.get_json() if request.is_json else request.form.to_dict()
        
        # Convert form data to API format
        blog_request = {
            "topic": form_data.get('topic', ''),
            "primary_keywords": _parse_keywords(form_data.get('primary_keywords', '')),
            "secondary_keywords": _parse_keywords(form_data.get('secondary_keywords', '')),
            "audience": form_data.get('audience', 'general'),
            "tone": form_data.get('tone', 'professional'),
            "writing_style": form_data.get('writing_style', 'informative'),
            "word_count": int(form_data.get('word_count', 1200)),
            "include_toc": form_data.get('include_toc') == 'true',
            "include_faqs": form_data.get('include_faqs') == 'true',
            "include_conclusion_cta": form_data.get('include_conclusion_cta') == 'true'
        }
        
        # Add personalization if provided
        if form_data.get('brand_name'):
            blog_request['personalization'] = {
                "brand_name": form_data.get('brand_name'),
                "company_website": form_data.get('company_website'),
                "author_bio": form_data.get('author_bio')
            }
            
            # Add social media
            social_media = {}
            for platform in ['instagram', 'twitter', 'linkedin', 'facebook', 'whatsapp']:
                if form_data.get(f'social_{platform}'):
                    social_media[platform] = form_data.get(f'social_{platform}')
            
            if social_media:
                blog_request['personalization']['social_media'] = social_media
        
        # Add external links if provided
        external_links = form_data.get('external_links', '').strip()
        if external_links:
            blog_request['external_links_needed'] = [link.strip() for link in external_links.split(',')]
        
        # Call FastAPI endpoint
        response = requests.post(
            f"{API_BASE_URL}/api/enhanced-blog",
            json=blog_request,
            timeout=120  # 2 minute timeout for AI generation
        )

        seo_score = (response.json().get('seo_guarantee', {}).get('achieved_score') or
             response.json().get('blog_data', {}).get('final_seo_score') or 0)

        if response.status_code == 200:
            result = response.json()
            
            # Save generated blog
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"blog_{timestamp}.json"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            # Convert markdown to HTML for display
            blog_content = result['blog_data']['content']
            html_content = markdown.markdown(blog_content, extensions=['extra', 'codehilite'])
            
            return jsonify({
                'success': True,
                'blog_data': result['blog_data'],
                'html_content': html_content,
                'analysis': result['content_analysis'],
                'user_actions': result['user_actions_needed'],
                'filename': filename
            })
        
        else:
            return jsonify({
                'success': False,
                'error': f"API Error: {response.status_code} - {response.text}"
            }), 500
            
    except requests.exceptions.Timeout:
        return jsonify({
            'success': False,
            'error': "Blog generation timed out. Please try again with a shorter word count."
        }), 500
        
    except requests.exceptions.ConnectionError:
        return jsonify({
            'success': False,
            'error': "Cannot connect to blog generator API. Make sure FastAPI server is running on port 8000."
        }), 500
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f"Unexpected error: {str(e)}"
        }), 500

@app.route('/load-sample/<sample_key>')
def load_sample(sample_key):
    """Load sample data for quick testing"""
    if sample_key in SAMPLE_DATA:
        return jsonify(SAMPLE_DATA[sample_key])
    else:
        return jsonify({'error': 'Sample not found'}), 404

@app.route('/preview/<filename>')
def preview_blog(filename):
    """Preview generated blog"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        blog_content = data['blog_data']['content']
        html_content = markdown.markdown(blog_content, extensions=['extra', 'codehilite'])
        
        return render_template('preview.html', 
                             blog_data=data['blog_data'],
                             html_content=Markup(html_content),
                             seo_score=seo_score,
                             analysis=data.get('content_analysis', {}),
                             filename=filename)
        
    except FileNotFoundError:
        flash('Blog file not found.', 'error')
        return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error loading blog: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_blog(filename):
    """Download blog content"""
    try:
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        blog_content = data['blog_data']['content']
        
        # Create downloadable content with copy instructions
        download_content = f"""<!-- 
SEO BLOG POST - COPY-READY FORMAT
Generated: {data['blog_data'].get('generated_at', '')}
Word Count: {data['blog_data'].get('word_count', 0)}
SEO Score: {data.get('seo_guarantee', {}).get('achieved_score') or data.get('blog_data', {}).get('final_seo_score', 0)}/100

STYLING PRESERVED:
- **Bold text** 
- *Italic text*
- # Headings
- [Links](url)
- > Callout boxes
- Emojis and formatting

INSTRUCTIONS:
1. Copy content below
2. Paste into your CMS/editor
3. Markdown formatting will be preserved
4. Add external links where [EXTERNAL: topic] appears
5. Verify all links work before publishing

-->

{blog_content}

<!-- 
METADATA FOR SEO:
Title: {data['blog_data']['metadata'].get('meta_title', '')}
Description: {data['blog_data']['metadata'].get('meta_description', '')}
Focus Keyword: {data['blog_data']['metadata'].get('focus_keyphrase', '')}
-->
"""
        
        from flask import Response
        return Response(
            download_content,
            mimetype='text/plain',
            headers={'Content-Disposition': f'attachment; filename=blog_content_{filename.replace(".json", ".md")}'})
        
    except Exception as e:
        flash(f'Error downloading blog: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api-status')
def api_status():
    """Check FastAPI backend status"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/health", timeout=5)
        if response.status_code == 200:
            return jsonify({'status': 'online', 'details': response.json()})
        else:
            return jsonify({'status': 'error', 'message': f'API returned {response.status_code}'})
    except requests.exceptions.ConnectionError:
        return jsonify({'status': 'offline', 'message': 'Cannot connect to FastAPI server'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

def _parse_keywords(keywords_string):
    """Parse comma-separated keywords"""
    if not keywords_string:
        return []
    return [kw.strip() for kw in keywords_string.split(',') if kw.strip()]

# Create HTML templates
# @app.before_first_request
def create_templates():
    """Create template files if they don't exist"""
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html template
    index_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SEO Blog Generator</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .gradient-bg { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .card-shadow { box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }
        .loading { display: none; }
        .result-section { display: none; }
        .api-status { position: fixed; top: 20px; right: 20px; z-index: 1000; }
    </style>
</head>
<body class="bg-light">
    <div class="api-status">
        <span class="badge bg-secondary" id="apiStatus">Checking API...</span>
    </div>

    <div class="container-fluid">
        <!-- Header -->
        <div class="row gradient-bg text-white py-4 mb-4">
            <div class="col-12 text-center">
                <h1><i class="fas fa-blog"></i> Enhanced SEO Blog Generator</h1>
                <p class="mb-0">Generate dynamic, styled blogs with AI-powered content structure</p>
            </div>
        </div>

        <div class="row">
            <!-- Form Section -->
            <div class="col-md-6">
                <div class="card card-shadow">
                    <div class="card-header bg-primary text-white">
                        <h4><i class="fas fa-edit"></i> Blog Configuration</h4>
                    </div>
                    <div class="card-body">
                        <!-- Quick Load Samples -->
                        <div class="mb-3">
                            <label class="form-label">Quick Start:</label>
                            <div class="btn-group w-100" role="group">
                                <button type="button" class="btn btn-outline-success" onclick="loadSample('dubai_real_estate')">
                                    Dubai Real Estate
                                </button>
                                <button type="button" class="btn btn-outline-info" onclick="loadSample('tech_saas')">
                                    Tech SaaS
                                </button>
                            </div>
                        </div>

                        <form id="blogForm">
                            <!-- Basic Info -->
                            <div class="row">
                                <div class="col-12 mb-3">
                                    <label class="form-label">Topic *</label>
                                    <input type="text" class="form-control" name="topic" placeholder="e.g., Best Dubai Real Estate Agents for Off-Plan Properties" required>
                                </div>
                            </div>

                            <div class="row">
                                <div class="col-md-6 mb-3">
                                    <label class="form-label">Primary Keywords *</label>
                                    <textarea class="form-control" name="primary_keywords" rows="2" placeholder="Dubai real estate agents, off-plan properties Dubai" required></textarea>
                                    <small class="text-muted">Comma-separated</small>
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label class="form-label">Secondary Keywords</label>
                                    <textarea class="form-control" name="secondary_keywords" rows="2" placeholder="property investment, RERA licensed"></textarea>
                                    <small class="text-muted">Comma-separated</small>
                                </div>
                            </div>

                            <!-- Content Settings -->
                            <div class="row">
                                <div class="col-md-4 mb-3">
                                    <label class="form-label">Audience</label>
                                    <select class="form-control" name="audience">
                                        <option value="general">General</option>
                                        <option value="property investors and buyers">Property Investors</option>
                                        <option value="startup founders and team leads">Startup Founders</option>
                                        <option value="small business owners">Small Business</option>
                                        <option value="marketing professionals">Marketing Pros</option>
                                    </select>
                                </div>
                                <div class="col-md-4 mb-3">
                                    <label class="form-label">Tone</label>
                                    <select class="form-control" name="tone">
                                        <option value="professional">Professional</option>
                                        <option value="friendly">Friendly</option>
                                        <option value="authoritative">Authoritative</option>
                                        <option value="casual">Casual</option>
                                        <option value="expert">Expert</option>
                                    </select>
                                </div>
                                <div class="col-md-4 mb-3">
                                    <label class="form-label">Writing Style</label>
                                    <select class="form-control" name="writing_style">
                                        <option value="informative">Informative</option>
                                        <option value="how-to">How-To Guide</option>
                                        <option value="listicle">Listicle</option>
                                        <option value="comparison">Comparison</option>
                                        <option value="storytelling">Storytelling</option>
                                    </select>
                                </div>
                            </div>

                            <div class="row">
                                <div class="col-md-6 mb-3">
                                    <label class="form-label">Word Count</label>
                                    <input type="number" class="form-control" name="word_count" value="1200" min="500" max="3000">
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label class="form-label">Features</label>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" name="include_toc" value="true" checked>
                                        <label class="form-check-label">Table of Contents</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" name="include_faqs" value="true" checked>
                                        <label class="form-check-label">FAQ Section</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" name="include_conclusion_cta" value="true" checked>
                                        <label class="form-check-label">Conclusion CTA</label>
                                    </div>
                                </div>
                            </div>

                            <!-- Personalization -->
                            <h5 class="mt-4 mb-3"><i class="fas fa-user-cog"></i> Personalization (Optional)</h5>
                            
                            <div class="row">
                                <div class="col-md-6 mb-3">
                                    <label class="form-label">Brand Name</label>
                                    <input type="text" class="form-control" name="brand_name" placeholder="OffPlan Market">
                                </div>
                                <div class="col-md-6 mb-3">
                                    <label class="form-label">Company Website</label>
                                    <input type="url" class="form-control" name="company_website" placeholder="https://offplan.market">
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">Author Bio</label>
                                <textarea class="form-control" name="author_bio" rows="2" placeholder="Property investment specialist with 8+ years experience..."></textarea>
                            </div>

                            <!-- Social Media -->
                            <h6><i class="fab fa-twitter"></i> Social Media Handles</h6>
                            <div class="row">
                                <div class="col-md-6 mb-2">
                                    <div class="input-group">
                                        <span class="input-group-text"><i class="fab fa-instagram"></i></span>
                                        <input type="text" class="form-control" name="social_instagram" placeholder="username">
                                    </div>
                                </div>
                                <div class="col-md-6 mb-2">
                                    <div class="input-group">
                                        <span class="input-group-text"><i class="fab fa-twitter"></i></span>
                                        <input type="text" class="form-control" name="social_twitter" placeholder="username">
                                    </div>
                                </div>
                                <div class="col-md-6 mb-2">
                                    <div class="input-group">
                                        <span class="input-group-text"><i class="fab fa-linkedin"></i></span>
                                        <input type="text" class="form-control" name="social_linkedin" placeholder="company-name">
                                    </div>
                                </div>
                                <div class="col-md-6 mb-2">
                                    <div class="input-group">
                                        <span class="input-group-text"><i class="fab fa-whatsapp"></i></span>
                                        <input type="text" class="form-control" name="social_whatsapp" placeholder="+971501234567">
                                    </div>
                                </div>
                            </div>

                            <div class="mb-3">
                                <label class="form-label">External Links Needed</label>
                                <textarea class="form-control" name="external_links" rows="2" placeholder="RERA official website, Dubai Land Department statistics"></textarea>
                                <small class="text-muted">Comma-separated topics that need external references</small>
                            </div>

                            <button type="submit" class="btn btn-primary btn-lg w-100">
                                <i class="fas fa-magic"></i> Generate Enhanced Blog
                            </button>
                        </form>

                        <!-- Loading State -->
                        <div class="loading text-center py-4">
                            <div class="spinner-border text-primary" role="status">
                                <span class="visually-hidden">Loading...</span>
                            </div>
                            <p class="mt-2">Generating your enhanced blog content...</p>
                            <small class="text-muted">This may take 1-2 minutes</small>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Results Section -->
            <div class="col-md-6">
                <div class="result-section">
                    <div class="card card-shadow">
                        <div class="card-header bg-success text-white d-flex justify-content-between align-items-center">
                            <h4><i class="fas fa-check-circle"></i> Generated Blog</h4>
                            <div>
                                <button class="btn btn-light btn-sm" onclick="copyToClipboard()">
                                    <i class="fas fa-copy"></i> Copy
                                </button>
                                <button class="btn btn-light btn-sm" onclick="downloadBlog()">
                                    <i class="fas fa-download"></i> Download
                                </button>
                            </div>
                        </div>
                        <div class="card-body">
                            <!-- Analysis Cards -->
                            <div class="row mb-3" id="analysisCards">
                                <!-- Analysis cards will be populated here -->
                            </div>

                            <!-- Blog Preview -->
                            <div class="blog-preview" id="blogPreview" style="max-height: 600px; overflow-y: auto; border: 1px solid #ddd; padding: 20px; background: white;">
                                <!-- Blog content will be displayed here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/js/bootstrap.bundle.min.js"></script>
    <script>
        let currentBlogData = null;
        let currentFilename = null;

        // Check API status
        checkApiStatus();

        function checkApiStatus() {
            fetch('/api-status')
                .then(response => response.json())
                .then(data => {
                    const statusElement = document.getElementById('apiStatus');
                    if (data.status === 'online') {
                        statusElement.className = 'badge bg-success';
                        statusElement.textContent = 'API Online';
                    } else {
                        statusElement.className = 'badge bg-danger';
                        statusElement.textContent = 'API Offline';
                    }
                })
                .catch(() => {
                    document.getElementById('apiStatus').className = 'badge bg-danger';
                    document.getElementById('apiStatus').textContent = 'API Error';
                });
        }

        function loadSample(sampleKey) {
            fetch(`/load-sample/${sampleKey}`)
                .then(response => response.json())
                .then(data => {
                    // Populate form with sample data
                    document.querySelector('[name="topic"]').value = data.topic;
                    document.querySelector('[name="primary_keywords"]').value = data.primary_keywords.join(', ');
                    document.querySelector('[name="secondary_keywords"]').value = data.secondary_keywords.join(', ');
                    document.querySelector('[name="audience"]').value = data.audience;
                    document.querySelector('[name="tone"]').value = data.tone;
                    document.querySelector('[name="writing_style"]').value = data.writing_style;
                    document.querySelector('[name="word_count"]').value = data.word_count;

                    if (data.personalization) {
                        document.querySelector('[name="brand_name"]').value = data.personalization.brand_name || '';
                        document.querySelector('[name="company_website"]').value = data.personalization.company_website || '';
                        document.querySelector('[name="author_bio"]').value = data.personalization.author_bio || '';

                        if (data.personalization.social_media) {
                            const social = data.personalization.social_media;
                            document.querySelector('[name="social_instagram"]').value = social.instagram || '';
                            document.querySelector('[name="social_twitter"]').value = social.twitter || '';
                            document.querySelector('[name="social_linkedin"]').value = social.linkedin || '';
                            document.querySelector('[name="social_whatsapp"]').value = social.whatsapp || '';
                        }
                    }
                })
                .catch(error => alert('Error loading sample: ' + error));
        }

        document.getElementById('blogForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const jsonData = {};
            
            for (let [key, value] of formData.entries()) {
                jsonData[key] = value;
            }

            // Show loading state
            document.querySelector('.loading').style.display = 'block';
            this.style.display = 'none';

            fetch('/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(jsonData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    currentBlogData = data;
                    currentFilename = data.filename;
                    displayResults(data);
                } else {
                    alert('Error: ' + data.error);
                }
            })
            .catch(error => {
                alert('Error: ' + error);
            })
            .finally(() => {
                document.querySelector('.loading').style.display = 'none';
                document.getElementById('blogForm').style.display = 'block';
            });
        });

        function displayResults(data) {
            // Show results section
            document.querySelector('.result-section').style.display = 'block';

            // Display analysis cards
            const analysisHtml = `
                <div class="col-md-3">
                    <div class="card bg-primary text-white">
                        <div class="card-body text-center">
                            <h5>${data.blog_data.word_count}</h5>
                            <small>Words</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-success text-white">
                        <div class="card-body text-center">
                            <h5>${data.seo_guarantee?.achieved_score || data.blog_data?.final_seo_score || 0}/100</h5>
                            <small>SEO Score</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-info text-white">
                        <div class="card-body text-center">
                            <h5>${data.analysis.readability_score.readability_level}</h5>
                            <small>Readability</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-warning text-white">
                        <div class="card-body text-center">
                            <h5>${Math.round(data.analysis.styling_completeness.styling_score)}%</h5>
                            <small>Styling</small>
                        </div>
                    </div>
                </div>
            `;
            document.getElementById('analysisCards').innerHTML = analysisHtml;

            // Display blog content
            document.getElementById('blogPreview').innerHTML = data.html_content;

            // Scroll to results
            document.querySelector('.result-section').scrollIntoView({ behavior: 'smooth' });
        }

        function copyToClipboard() {
            if (currentBlogData) {
                const content = currentBlogData.blog_data.content;
                navigator.clipboard.writeText(content).then(() => {
                    alert('Blog content copied to clipboard!');
                }).catch(() => {
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = content;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    alert('Blog content copied to clipboard!');
                });
            }
        }

        function downloadBlog() {
            if (currentFilename) {
                window.open(`/download/${currentFilename}`, '_blank');
            }
        }

        // Auto-refresh API status every 30 seconds
        setInterval(checkApiStatus, 30000);
    </script>
</body>
</html>'''

    # Create preview.html template
    preview_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Blog Preview - SEO Blog Generator</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.0/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .blog-content { 
            line-height: 1.8; 
            font-size: 16px; 
        }
        .blog-content h1 { color: #2c3e50; margin-bottom: 1rem; }
        .blog-content h2 { color: #34495e; margin: 2rem 0 1rem 0; }
        .blog-content h3 { color: #7f8c8d; margin: 1.5rem 0 0.8rem 0; }
        .blog-content blockquote {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 1rem;
            margin: 1rem 0;
        }
        .stats-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .copy-button {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
        }
    </style>
</head>
<body class="bg-light">
    <div class="container-fluid py-4">
        <div class="row">
            <div class="col-md-3">
                <!-- Blog Statistics -->
                <div class="card stats-card mb-4">
                    <div class="card-body text-center">
                        <h5>Blog Statistics</h5>
                        <hr class="bg-white">
                        <div class="row">
                            <div class="col-6">
                                <h3>{{ blog_data.word_count }}</h3>
                                <small>Words</small>
                            </div>
                            <div class="col-6">
                                <h3>{{ blog_data.final_seo_score | default(0) }}/100</h3>
                                <small>SEO Score</small>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- SEO Metadata -->
                <div class="card mb-4">
                    <div class="card-header bg-primary text-white">
                        <h6><i class="fas fa-search"></i> SEO Metadata</h6>
                    </div>
                    <div class="card-body">
                        <div class="mb-3">
                            <label class="form-label fw-bold">Meta Title:</label>
                            <p class="small">{{ blog_data.metadata.meta_title }}</p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label fw-bold">Meta Description:</label>
                            <p class="small">{{ blog_data.metadata.meta_description }}</p>
                        </div>
                        <div class="mb-3">
                            <label class="form-label fw-bold">Focus Keyword:</label>
                            <p class="small">{{ blog_data.metadata.focus_keyphrase }}</p>
                        </div>
                    </div>
                </div>

                <!-- Actions -->
                <div class="card">
                    <div class="card-body">
                        <button class="btn btn-success btn-block mb-2" onclick="copyBlog()">
                            <i class="fas fa-copy"></i> Copy Content
                        </button>
                        <a href="/download/{{ filename }}" class="btn btn-primary btn-block mb-2">
                            <i class="fas fa-download"></i> Download
                        </a>
                        <a href="/" class="btn btn-secondary btn-block">
                            <i class="fas fa-arrow-left"></i> Back to Generator
                        </a>
                    </div>
                </div>
            </div>

            <div class="col-md-9">
                <!-- Blog Content -->
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5><i class="fas fa-file-alt"></i> Generated Blog Content</h5>
                        <small class="text-muted">Generated: {{ blog_data.generated_at }}</small>
                    </div>
                    <div class="card-body blog-content">
                        {{ html_content|safe }}
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Copy Button -->
    <button class="btn btn-success copy-button" onclick="copyBlog()">
        <i class="fas fa-copy"></i>
    </button>

    <script>
        function copyBlog() {
            const blogContent = `{{ blog_data.content|replace('\n', '\\n')|replace('\r', '')|replace('"', '\\"') }}`;
            
            navigator.clipboard.writeText(blogContent).then(() => {
                // Show success feedback
                const btn = document.querySelector('.copy-button');
                const originalContent = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-check"></i>';
                btn.classList.remove('btn-success');
                btn.classList.add('btn-success');
                
                setTimeout(() => {
                    btn.innerHTML = originalContent;
                }, 2000);
                
                alert('Blog content copied to clipboard!');
            }).catch(() => {
                alert('Copy failed. Please use the download option instead.');
            });
        }
    </script>
</body>
</html>'''

    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_template)
    
    with open(os.path.join(templates_dir, 'preview.html'), 'w', encoding='utf-8') as f:
        f.write(preview_template)

if __name__ == '__main__':
    create_templates()
    print("🚀 SEO Blog Generator - Flask UI")
    print("=" * 50)
    print("\n📋 Setup Instructions:")
    print("1. Make sure your FastAPI server is running on port 8000")
    print("2. Install Flask dependencies:")
    print("   pip install flask requests markdown")
    print("\n🌐 Endpoints:")
    print("  - Main UI: http://127.0.0.1:5000")
    print("  - API Status: http://127.0.0.1:5000/api-status")
    print("  - Preview: http://127.0.0.1:5000/preview/<filename>")
    print("\n🔧 Features:")
    print("  - ✅ Form-based blog generation")
    print("  - ✅ Real-time API status monitoring")
    print("  - ✅ Sample data quick-load")
    print("  - ✅ Live blog preview with styling")
    print("  - ✅ Copy to clipboard functionality")
    print("  - ✅ Download as markdown file")
    print("  - ✅ SEO analysis display")
    print("  - ✅ Responsive design")
    
    print("\n💡 Usage:")
    print("1. Load sample data or fill form manually")
    print("2. Click 'Generate Enhanced Blog'")
    print("3. View results with SEO analysis")
    print("4. Copy content or download file")
    
    print("\n🔗 Integration:")
    print("  - Communicates with FastAPI backend via HTTP")
    print("  - Handles timeouts and connection errors")
    print("  - Saves generated blogs locally")
    print("  - Markdown to HTML conversion for preview")
    
    print("\n" + "=" * 50)
    print("🎯 Starting Flask UI on http://127.0.0.1:5000")
    print("📡 Expecting FastAPI backend on http://127.0.0.1:8000")

    app.run(host='0.0.0.0', port=5000, debug=True)