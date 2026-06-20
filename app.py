"""
GRAHAM INVESTMENT AGENT — Web App (Visual Overhaul)
====================================================
Same logic as original. Radically different skin.
"""
# --- SQLITE PATCH FOR STREAMLIT CLOUD ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ----------------------------------------

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import streamlit as st
from google import genai
from google.genai import types
import chromadb
import pymupdf
import yfinance as yf
import json

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Graham Investment Agent",
    page_icon="📈",
    layout="centered"
)

# ──────────────────────────────────────────────
# CUSTOM CSS — the entire visual overhaul
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ═══════════════════════════════════════════════
   ANIMATED AURORA GRADIENT BACKGROUND
   ═══════════════════════════════════════════════ */
@keyframes aurora {
    0%   { background-position: 0% 50%; }
    25%  { background-position: 50% 100%; }
    50%  { background-position: 100% 50%; }
    75%  { background-position: 50% 0%; }
    100% { background-position: 0% 50%; }
}

/* Root app background */
.stApp {
    background: linear-gradient(
        -45deg,
        #0f0c29,
        #1a1a40,
        #302b63,
        #24243e,
        #0f0c29,
        #1b1145,
        #0d2137,
        #0a1628
    );
    background-size: 400% 400%;
    animation: aurora 20s ease infinite;
}

/* ═══════════════════════════════════════════════
   DEEP SPACE STARFIELD (Moving & Twinkling)
   ═══════════════════════════════════════════════ */
@keyframes starDrift {
    0% { 
        background-position: 0 0; 
        opacity: 0.4; 
    }
    50% { 
        opacity: 0.9; /* Stars twinkle brighter mid-cycle */
    }
    100% { 
        background-position: 0 -200px; /* Stars drift slowly upwards */
        opacity: 0.4; 
    }
}

.stApp::before {
    content: '';
    position: fixed;
    top: -200px; left: 0; right: 0; bottom: 0; /* Overshoot top to allow seamless scrolling */
    /* Generate hundreds of stars using radial gradients */
    background-image:
        radial-gradient(1px 1px at 15% 25%, rgba(255, 255, 255, 1) 50%, transparent),
        radial-gradient(2px 2px at 35% 65%, rgba(0, 245, 212, 0.8) 50%, transparent), /* Cyan stars */
        radial-gradient(1.5px 1.5px at 55% 15%, rgba(255, 255, 255, 0.9) 50%, transparent),
        radial-gradient(1px 1px at 75% 85%, rgba(200, 210, 230, 0.8) 50%, transparent),
        radial-gradient(2px 2px at 85% 35%, rgba(255, 255, 255, 1) 50%, transparent),
        radial-gradient(1px 1px at 25% 75%, rgba(0, 245, 212, 0.6) 50%, transparent),
        radial-gradient(1.5px 1.5px at 95% 55%, rgba(255, 255, 255, 0.9) 50%, transparent),
        radial-gradient(2px 2px at 5% 45%, rgba(200, 210, 230, 0.8) 50%, transparent),
        radial-gradient(1px 1px at 45% 95%, rgba(255, 255, 255, 1) 50%, transparent);
    background-size: 150px 150px; /* Creates a repeating grid of the stars above */
    pointer-events: none;
    z-index: 0;
    /* 12s animation makes it drift slowly, alternate makes it smoothly reverse */
    animation: starDrift 12s ease-in-out infinite alternate;
}

/* ═══════════════════════════════════════════════
   TYPOGRAPHY & TITLE — NEON GLOW
   ═══════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');

.stApp, .stApp * {
    font-family: 'Inter', sans-serif !important;
}

/* Main title glow */
.stApp h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 2.6rem !important;
    background: linear-gradient(135deg, #00f5d4, #7bf1a8, #fee440, #f15bb5, #9b5de5);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: aurora 6s ease infinite;
    text-shadow: none;
    filter: drop-shadow(0 0 30px rgba(0, 245, 212, 0.3));
    padding-bottom: 4px;
}

/* Caption */
.stApp .stCaption, .stApp [data-testid="stCaptionContainer"] p {
    color: rgba(200, 210, 230, 0.6) !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px;
}

/* ═══════════════════════════════════════════════
   GLASSMORPHISM CHAT BUBBLES
   ═══════════════════════════════════════════════ */

/* Chat message containers */
[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(20px) !important;
    -webkit-backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
    margin-bottom: 12px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
}

[data-testid="stChatMessage"]:hover {
    background: rgba(255, 255, 255, 0.07) !important;
    border-color: rgba(120, 200, 255, 0.15) !important;
    box-shadow:
        0 8px 32px rgba(0, 0, 0, 0.3),
        0 0 20px rgba(120, 200, 255, 0.05),
        inset 0 1px 0 rgba(255, 255, 255, 0.08) !important;
    transform: translateY(-1px);
}

/* Chat text color */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span {
    color: rgba(230, 235, 245, 0.92) !important;
    line-height: 1.7 !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatMessage"] strong {
    color: #7bf1a8 !important;
}

[data-testid="stChatMessage"] code {
    background: rgba(0, 245, 212, 0.1) !important;
    color: #00f5d4 !important;
    border-radius: 4px !important;
    padding: 2px 6px !important;
}

/* User avatar ring */
[data-testid="stChatMessage"] [data-testid="stAvatar"] {
    border: 2px solid rgba(0, 245, 212, 0.4) !important;
    border-radius: 50% !important;
    box-shadow: 0 0 12px rgba(0, 245, 212, 0.15) !important;
}

/* ═══════════════════════════════════════════════
   CHAT INPUT — GLOWING BAR
   ═══════════════════════════════════════════════ */
[data-testid="stChatInput"],
[data-testid="stChatInputContainer"] {
    background: transparent !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 14px !important;
    color: #e6ebf5 !important;
    font-size: 0.95rem !important;
    padding: 14px 18px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInputContainer"] textarea:focus {
    border-color: rgba(0, 245, 212, 0.5) !important;
    box-shadow:
        0 0 20px rgba(0, 245, 212, 0.1),
        0 0 40px rgba(0, 245, 212, 0.05),
        inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    outline: none !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(200, 210, 230, 0.35) !important;
}

/* Send button */
[data-testid="stChatInput"] button,
[data-testid="stChatInputContainer"] button {
    background: linear-gradient(135deg, #00f5d4, #9b5de5) !important;
    border: none !important;
    border-radius: 10px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stChatInput"] button:hover,
[data-testid="stChatInputContainer"] button:hover {
    filter: brightness(1.2) !important;
    box-shadow: 0 0 20px rgba(0, 245, 212, 0.3) !important;
}

/* ═══════════════════════════════════════════════
   NEW CHAT BUTTON — PILL STYLE
   ═══════════════════════════════════════════════ */
.stButton > button {
    background: rgba(255, 255, 255, 0.06) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 50px !important;
    color: rgba(200, 210, 230, 0.8) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    padding: 8px 24px !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: rgba(0, 245, 212, 0.12) !important;
    border-color: rgba(0, 245, 212, 0.4) !important;
    color: #00f5d4 !important;
    box-shadow: 0 0 25px rgba(0, 245, 212, 0.1) !important;
    transform: translateY(-2px) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* ═══════════════════════════════════════════════
   SPINNER — CUSTOM COLOR
   ═══════════════════════════════════════════════ */
.stSpinner > div {
    border-top-color: #00f5d4 !important;
}

[data-testid="stSpinnerContainer"] {
    color: rgba(200, 210, 230, 0.6) !important;
}

/* ═══════════════════════════════════════════════
   SCROLLBAR — THIN & THEMED
   ═══════════════════════════════════════════════ */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}

::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.2);
}

::-webkit-scrollbar-thumb {
    background: rgba(0, 245, 212, 0.2);
    border-radius: 3px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 245, 212, 0.4);
}

/* ═══════════════════════════════════════════════
   SIDEBAR (if ever used)
   ═══════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.95) !important;
    backdrop-filter: blur(20px) !important;
}

/* ═══════════════════════════════════════════════
   HIDE STREAMLIT DEFAULTS
   ═══════════════════════════════════════════════ */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* ═══════════════════════════════════════════════
   AGGRESSIVE TRANSPARENT INPUT DOCK
   ═══════════════════════════════════════════════ */
[data-testid="stBottom"], 
[data-testid="stBottom"] > div,
[data-testid="stBottom"] [data-testid="stVerticalBlock"] {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
}

/* Ensure the main view wrapper isn't creating a solid block behind it */
[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

/* ═══════════════════════════════════════════════
   WELCOME CARD (Terminal Dashboard)
   ═══════════════════════════════════════════════ */
.welcome-card {
    background: rgba(15, 12, 41, 0.4);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0, 245, 212, 0.15);
    border-radius: 8px; /* Sharper corners */
    padding: 2.5rem 2rem;
    text-align: center;
    margin: 2rem auto;
    max-width: 550px;
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), inset 0 0 20px rgba(0, 245, 212, 0.05);
}

.welcome-card h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.2rem;
    color: #00f5d4; /* Neon terminal green/cyan */
    margin-bottom: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
}

.welcome-card p {
    color: rgba(200, 210, 230, 0.6);
    font-size: 0.9rem;
    font-family: 'Inter', sans-serif !important;
    line-height: 1.6;
    margin: 0;
}

.welcome-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    justify-content: center;
    margin-top: 1.8rem;
}

.welcome-pill {
    background: rgba(0, 245, 212, 0.05);
    border: 1px solid rgba(0, 245, 212, 0.25);
    border-radius: 4px; /* Tech/Command box look instead of round pill */
    padding: 8px 16px;
    color: #00f5d4;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    transition: all 0.2s ease;
}

.welcome-pill:hover {
    background: rgba(0, 245, 212, 0.15);
    border-color: rgba(0, 245, 212, 0.5);
    box-shadow: 0 0 15px rgba(0, 245, 212, 0.2);
    cursor: default;
}

/* ═══════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════ */
@media (max-width: 768px) {
    .stApp h1 {
        font-size: 1.8rem !important;
    }
    .welcome-card {
        margin: 1rem;
        padding: 1.5rem 1.2rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.title("🏛️ AlphaConsensus Terminal")
st.caption("Quantitative Multi-Agent Investment Committee. Operating on Graham, Greenblatt, and Dorsey frameworks.")

# New Chat button
if st.button("🔄 Reset Terminal"):
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.rerun()

# ──────────────────────────────────────────────
# LOAD BOOKS INTO CHROMADB (runs once, cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_books():
    books = {
        "Graham": "The Intelligent Investor.pdf",
        "Greenblatt": "The Little Book That Still Beats the Market.pdf",
        "Dorsey": "The Five Rules for Successful Stock Investing.pdf"
    }
    
    chroma_client = chromadb.EphemeralClient()
    # Use get_or_create to avoid duplication errors on hot reloads
    collection = chroma_client.get_or_create_collection("investment_committee")
    
    # Check if we already loaded them
    if collection.count() > 0:
        return collection

    for author, filename in books.items():
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found.")
            continue
            
        doc = pymupdf.open(filename)
        full_text = "\n".join(page.get_text() for page in doc)
        doc.close()

        raw_paragraphs = full_text.split("\n\n")
        chunks = []
        current = ""
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) < 1500:
                current = current + "\n\n" + para if current else para
            else:
                if len(current) >= 100:
                    chunks.append(current)
                current = para
        if current and len(current) >= 100:
            chunks.append(current)

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            collection.add(
                documents=batch,
                metadatas=[{"author": author} for _ in batch],
                ids=[f"{author}_chunk_{j}" for j in range(i, i + len(batch))]
            )
            
    return collection

collection = load_books()

# ──────────────────────────────────────────────
# TOOLS (unchanged)
# ──────────────────────────────────────────────

def search_book(query: str) -> dict:
    """Search the combined knowledge base of Graham, Greenblatt, and Dorsey.
    Use this when you need specific philosophical frameworks, formulas, or rules 
    from any of the three investment authors.
    
    Args:
        query: What to search for, e.g. "magic formula return on capital" or "economic moat"
    """
    sem_results = collection.query(query_texts=[query], n_results=5)
    
    if not sem_results["documents"][0]:
        return {"error": "No relevant passages found."}
        
    sem_docs = sem_results["documents"][0]
    sem_meta = sem_results["metadatas"][0]
    sem_dists = sem_results["distances"][0]

    formatted = []
    for text, meta, dist in zip(sem_docs, sem_meta, sem_dists):
        author = meta.get("author", "Unknown")
        # Prepend the author so the LLM explicitly knows the source
        formatted.append(f"[Source: {author} | Relevance: {1-dist:.2f}]:\n{text}")

    return {"passages": "\n\n".join(formatted)}


def get_stock_data(ticker: str) -> dict:
    """Get real financial data for a stock using its ticker symbol.
    Use this when the user asks about a specific company's financials,
    P/E ratio, earnings, dividend yield, book value, or any fundamental data.

    For Indian stocks on NSE, append .NS (e.g., TCS.NS, INFY.NS, RELIANCE.NS).
    For US stocks, use plain ticker (e.g., AAPL for Apple, MSFT for Microsoft).
    For Mahindra, use MAHINDRA (the function resolves it automatically).

    Args:
        ticker: Stock ticker symbol, e.g. AAPL, MSFT, TCS.NS, MAHINDRA, RELIANCE.NS
    """
    TICKER_ALIASES = {
        "MAHINDRA": "M&M.NS",
        "MAHINDRA.NS": "M&M.NS",
        "M_M.NS": "M&M.NS",
        "MM.NS": "M&M.NS",
        "MNM.NS": "M&M.NS",
        "M&M": "M&M.NS",
        "L&T": "LT.NS",
        "L&T.NS": "LT.NS",
        "L_T.NS": "LT.NS",
    }
    resolved = TICKER_ALIASES.get(ticker.upper(), ticker)

    try:
        stock = yf.Ticker(resolved)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            return {"error": f"No data found for '{ticker}' (tried '{resolved}'). Check the ticker. Indian stocks need .NS suffix (e.g., TCS.NS, INFY.NS)."}

        return {
            "symbol": info.get("symbol"),
            "name": info.get("longName") or info.get("shortName"),
            "sector": info.get("sector"),
            "currency": info.get("currency"),
            "current_price": info.get("regularMarketPrice") or info.get("currentPrice"),
            "market_cap": info.get("marketCap"),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "book_value": info.get("bookValue"),
            "eps": info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "profit_margin": info.get("profitMargins"),
            "return_on_equity": info.get("returnOnEquity"),
            "revenue": info.get("totalRevenue"),
            "debt_to_equity": info.get("debtToEquity"),
            "current_ratio": info.get("currentRatio"),
            "52_week_high": info.get("fiftyTwoWeekHigh"),
            "52_week_low": info.get("fiftyTwoWeekLow"),
        }
    except Exception as e:
        return {"error": f"Failed to fetch data for '{ticker}': {str(e)}"}


def calculator(expression: str) -> dict:
    """Evaluate a math expression. Use for any calculation:
    ratios, percentages, comparisons, margin of safety computations, etc.

    Args:
        expression: A Python math expression, e.g. "45000 / 1200" or "(52.3 - 41.8) / 52.3 * 100"
    """
    try:
        result = eval(expression)
        return {"expression": expression, "result": round(result, 4)}
    except Exception as e:
        return {"error": f"Could not evaluate '{expression}': {str(e)}"}


tool_functions = {
    "search_book": search_book,
    "get_stock_data": get_stock_data,
    "calculator": calculator,
}

# ──────────────────────────────────────────────
# SYSTEM PROMPT & AGENT
# ──────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a highly structured Quantitative Investment Committee acting as a single agent. 

Your knowledge base consists of three frameworks:
1. Benjamin Graham (Defensive Value, Margin of Safety, P/B, P/E)
2. Joel Greenblatt (The Magic Formula, Return on Capital, Earnings Yield)
3. Pat Dorsey (Economic Moats, Consistent FCF, ROE > 15%, Low Debt)

You have three tools:
1. search_books — queries the texts of Graham, Greenblatt, and Dorsey.
2. get_stock_data — pulls live fundamental data.
3. calculator — evaluates mathematical expressions.

CRITICAL INSTRUCTIONS FOR STOCK ANALYSIS:
When the user asks you to evaluate a stock, you MUST NOT write a generic summary. You must execute a "Three-Factor Committee Analysis" using Markdown tables and provide a definitive YES/NO verdict.

Format your response EXACTLY like this:

### 1. Live Fundamentals
[Render a clean Markdown table with the stock's current price, P/E, Forward P/E, P/B, ROE, Debt/Equity, and Dividend Yield]

### 2. The Committee Verdict
Evaluate the data against the three frameworks. If you are unsure of a specific rule, use the search_books tool.

* **Graham's Verdict:** [Pass/Fail] - [One sentence justification based on Margin of Safety/Valuation]
* **Greenblatt's Verdict:** [Pass/Fail] - [One sentence justification based on Earnings Yield/Capital Efficiency]
* **Dorsey's Verdict:** [Pass/Fail] - [One sentence justification based on inferred Moat/Financial Health]

### 3. Final Decision
**[YES or NO]** [If 2 out of 3 Pass, it is a YES. If not, it is a NO. Provide a brief, blunt, one-paragraph explanation of the final ruling.]
"""

TOOLS = [search_book, get_stock_data, calculator]


def agent_turn(user_message):
    """Create a fresh client every turn."""
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    history = st.session_state.get("chat_history", [])

    chat = client.chats.create(
        model="gemini-2.5-flash-lite",
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=TOOLS,
        ),
        history=history,
    )

    response = chat.send_message(user_message)

    while response.function_calls:
        function_responses = []
        for fc in response.function_calls:
            if fc.name in tool_functions:
                result = tool_functions[fc.name](**fc.args)
            else:
                result = {"error": f"Unknown tool: {fc.name}"}
            function_responses.append(
                types.Part.from_function_response(name=fc.name, response=result)
            )
        response = chat.send_message(function_responses)

    st.session_state.chat_history = chat.get_history()
    return response.text

# ──────────────────────────────────────────────
# CHAT UI
# ──────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Welcome card when empty
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h2>SYSTEM INITIALIZED</h2>
        <p>AlphaConsensus engine online. Awaiting ticker input or framework query.</p>
        <div class="welcome-pills">
            <span class="welcome-pill">> ANALYZE AAPL</span>
            <span class="welcome-pill">> QUERY MOAT RULES</span>
            <span class="welcome-pill">> MAGIC FORMULA YIELD</span>
            <span class="welcome-pill">> GRAHAM CRITERIA</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Define your avatars
USER_AVATAR = "👤"
AGENT_AVATAR = "📈"

# Display past messages
for msg in st.session_state.messages:
    # Assign the correct avatar based on the role
    avatar = USER_AVATAR if msg["role"] == "user" else AGENT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle new input
if prompt := st.chat_input("Ask about a stock, Graham's principles, or anything..."):
    # 1. Save user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Display user message in UI
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    # 3. Handle agent response
    with st.chat_message("assistant", avatar=AGENT_AVATAR):
        with st.spinner("Analyzing..."):
            try:
                # Generate and display the answer
                answer = agent_turn(prompt)
                st.markdown(answer)
                # Save agent message to state ONLY if successful
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except Exception as e:
                # Graceful error handling
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    st.error("⚠️ **API Rate Limit Exceeded.** The Gemini API has reached its limit. Please wait a moment and try again.")
                else:
                    st.error(f"🛑 **System Error:** Unable to process request. \n\n`{error_msg[:100]}...`")
