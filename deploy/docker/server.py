import math
import os
import sys
import time
from typing import List, Optional, Dict, Literal
from fastapi import FastAPI, HTTPException, Request, Query, Path, Depends
from fastapi.responses import StreamingResponse, RedirectResponse, PlainTextResponse, JSONResponse
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from redis import asyncio as aioredis

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utils import FilterType, load_config, setup_logging, verify_email_domain
from api import (
    handle_markdown_request,
    handle_llm_qa,
    handle_stream_crawl_request,
    handle_crawl_request,
    handle_deep_crawl_request,
    stream_results
)
from auth import create_access_token, get_token_dependency, TokenRequest  # Import from auth.py

__version__ = "0.2.6"

class CrawlRequest(BaseModel):
    url: str = Field(..., description="The URL to crawl")
    browser_config: Optional[Dict] = Field(default_factory=dict)
    crawler_config: Optional[Dict] = Field(default_factory=dict)

class DeepCrawlRequest(CrawlRequest):
    deep_crawl_strategy: Literal["BFS", "DFS", "BestFirst"] = Field(
        default="BFS", 
        description="The deep crawling strategy to use: BFS (Breadth-First Search), DFS (Depth-First Search), or BestFirst"
    )
    max_depth: int = Field(
        default=2, 
        ge=1, 
        le=5, 
        description="Maximum depth to crawl (1-5)"
    )
    include_external: bool = Field(
        default=False, 
        description="Whether to include external links (links to other domains)"
    )
    max_pages: Optional[int] = Field(
        default=math.inf, 
        ge=1, 
        le=500, 
        description="Maximum number of pages to crawl (1-500)"
    )
    score_threshold: Optional[float] = Field(
        default=-math.inf, 
        description="Minimum score for URLs to be crawled (only used with BestFirst strategy)"
    )
    keywords: Optional[List[str]] = Field(
        default=None, 
        description="Keywords for relevance scoring (only used with BestFirst strategy)"
    )
    keyword_weight: Optional[float] = Field(
        default=0.7, 
        ge=0.0, 
        le=1.0, 
        description="Weight for keyword relevance (only used with BestFirst strategy)"
    )

# Load configuration and setup
config = load_config()
setup_logging(config)

# Initialize Redis
redis = aioredis.from_url(config["redis"].get("uri", "redis://localhost"))

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[config["rate_limiting"]["default_limit"]],
    storage_uri=config["rate_limiting"]["storage_uri"]
)

app = FastAPI(
    title=config["app"]["title"],
    version=config["app"]["version"]
)

# Configure middleware
def setup_security_middleware(app, config):
    sec_config = config.get("security", {})
    if sec_config.get("enabled", False):
        if sec_config.get("https_redirect", False):
            app.add_middleware(HTTPSRedirectMiddleware)
        if sec_config.get("trusted_hosts", []) != ["*"]:
            app.add_middleware(TrustedHostMiddleware, allowed_hosts=sec_config["trusted_hosts"])

setup_security_middleware(app, config)

# Prometheus instrumentation
if config["observability"]["prometheus"]["enabled"]:
    Instrumentator().instrument(app).expose(app)

# Get token dependency based on config
token_dependency = get_token_dependency(config)

# Middleware for security headers
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    if config["security"]["enabled"]:
        response.headers.update(config["security"]["headers"])
    return response

# Token endpoint (always available, but usage depends on config)
@app.post("/token")
async def get_token(request_data: TokenRequest):
    if not verify_email_domain(request_data.email):
        raise HTTPException(status_code=400, detail="Invalid email domain")
    token = create_access_token({"sub": request_data.email})
    return {"email": request_data.email, "access_token": token, "token_type": "bearer"}

# Endpoints with conditional auth
@app.get("/md/{url:path}")
@limiter.limit(config["rate_limiting"]["default_limit"])
async def get_markdown(
    request: Request,
    url: str,
    f: FilterType = FilterType.FIT,
    q: Optional[str] = None,
    c: Optional[str] = "0",
    token_data: Optional[Dict] = Depends(token_dependency)
):
    result = await handle_markdown_request(url, f, q, c, config)
    return PlainTextResponse(result)

@app.get("/llm/{url:path}", description="URL should be without http/https prefix")
async def llm_endpoint(
    request: Request,
    url: str = Path(...),
    q: Optional[str] = Query(None),
    token_data: Optional[Dict] = Depends(token_dependency)
):
    if not q:
        raise HTTPException(status_code=400, detail="Query parameter 'q' is required")
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    try:
        answer = await handle_llm_qa(url, q, config)
        return JSONResponse({"answer": answer})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
async def get_schema():
    from crawl4ai import BrowserConfig, CrawlerRunConfig
    return {"browser": BrowserConfig().dump(), "crawler": CrawlerRunConfig().dump()}

@app.get(config["observability"]["health_check"]["endpoint"])
async def health():
    return {"status": "ok", "timestamp": time.time(), "version": __version__}

@app.get(config["observability"]["prometheus"]["endpoint"])
async def metrics():
    return RedirectResponse(url=config["observability"]["prometheus"]["endpoint"])

@app.post("/crawl")
@limiter.limit(config["rate_limiting"]["default_limit"])
async def crawl(
    request: Request,
    crawl_request: CrawlRequest,
    token_data: Optional[Dict] = Depends(token_dependency)
):
    if not crawl_request.urls:
        raise HTTPException(status_code=400, detail="At least one URL required")
    
    results = await handle_crawl_request(
        urls=crawl_request.urls,
        browser_config=crawl_request.browser_config,
        crawler_config=crawl_request.crawler_config,
        config=config
    )

    return JSONResponse(results)

@app.post("/crawl/stream")
@limiter.limit(config["rate_limiting"]["default_limit"])
async def crawl_stream(
    request: Request,
    crawl_request: CrawlRequest,
    token_data: Optional[Dict] = Depends(token_dependency)
):
    if not crawl_request.urls:
        raise HTTPException(status_code=400, detail="At least one URL required")

    crawler, results_gen = await handle_stream_crawl_request(
        urls=crawl_request.urls,
        browser_config=crawl_request.browser_config,
        crawler_config=crawl_request.crawler_config,
        config=config
    )

    return StreamingResponse(
        stream_results(crawler, results_gen),
        media_type='application/x-ndjson',
        headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'X-Stream-Status': 'active'}
    )

@app.post("/deep-crawl", description="Deep crawl a website with configurable strategy (BFS, DFS, or BestFirst)")
@limiter.limit(config["rate_limiting"]["default_limit"])
async def deep_crawl(
    request: Request,
    crawl_request: DeepCrawlRequest,
    token_data: Optional[Dict] = Depends(token_dependency)
):
    """Deep crawl endpoint that explores websites beyond a single page.
    
    This endpoint allows configuring the crawling strategy (BFS, DFS, or BestFirst),
    depth, and other parameters to control the crawl behavior.
    
    - BFS (Breadth-First Search): Explores all links at one depth before moving deeper
    - DFS (Depth-First Search): Explores as far down a branch as possible before backtracking
    - BestFirst: Prioritizes pages based on relevance to specified keywords
    """
    if not crawl_request.url:
        raise HTTPException(status_code=400, detail="A URL required")
    
    # Validate BestFirst strategy requirements
    if crawl_request.deep_crawl_strategy == "BestFirst" and not crawl_request.keywords:
        raise HTTPException(
            status_code=400, 
            detail="Keywords are required for BestFirst crawling strategy"
        )
    
    results = await handle_deep_crawl_request(
        urls=[crawl_request.url],
        browser_config=crawl_request.browser_config,
        crawler_config=crawl_request.crawler_config,
        deep_crawl_strategy=crawl_request.deep_crawl_strategy,
        max_depth=crawl_request.max_depth,
        include_external=crawl_request.include_external,
        config=config,
        max_pages=crawl_request.max_pages,
        score_threshold=crawl_request.score_threshold,
        keywords=crawl_request.keywords,
        keyword_weight=crawl_request.keyword_weight
    )

    return JSONResponse(results)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=config["app"]["host"],
        port=config["app"]["port"],
        reload=config["app"]["reload"],
        timeout_keep_alive=config["app"]["timeout_keep_alive"]
    )