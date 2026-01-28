"""
Agentic Memory Demo with Chainlit
==================================
Beautiful chat UI demonstrating cross-session memory with semantic-router.

Run with:
    cd scripts && chainlit run chainlit_demo.py -w --port 8000

Requirements:
    pip install chainlit requests
"""

import os
import time
import requests
import chainlit as cl
from datetime import datetime

# Configuration
ROUTER_URL = os.getenv("ROUTER_URL", "http://localhost:8801")
MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Demo users (for video recording - simple auth)
DEMO_USERS = {
    "yehudit": {"password": "demo", "display_name": "Yehudit"},
    "moti": {"password": "demo", "display_name": "Moti"},
    "alex": {"password": "demo", "display_name": "Alex"},
    "sarah": {"password": "demo", "display_name": "Sarah"},
    "demo": {"password": "demo", "display_name": "Demo User"},
}

# Metrics tracking (per-user)
metrics_store = {}


def get_user_metrics(user_id: str) -> dict:
    """Get or create metrics for a user."""
    if user_id not in metrics_store:
        metrics_store[user_id] = {
            "total_requests": 0,
            "memories_retrieved": 0,
            "memories_stored": 0,
            "latencies": [],
        }
    return metrics_store[user_id]


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Authenticate users for the demo."""
    username_lower = username.lower()
    if username_lower in DEMO_USERS:
        if DEMO_USERS[username_lower]["password"] == password:
            return cl.User(
                identifier=username_lower,
                metadata={"name": DEMO_USERS[username_lower]["display_name"]},
            )
    # Allow any user with password "demo" for flexibility
    if password == "demo":
        return cl.User(
            identifier=username.lower().replace(" ", "_"),
            metadata={"name": username},
        )
    return None


def update_metrics_display(user_id: str):
    """Create metrics display string for a user."""
    metrics = get_user_metrics(user_id)
    avg_latency = sum(metrics["latencies"][-10:]) / len(metrics["latencies"][-10:]) if metrics["latencies"] else 0
    return f"""
**📊 Your Session Metrics**
━━━━━━━━━━━━━━━━━━━━━━━
🔄 Requests: {metrics["total_requests"]}
🧠 Memories Retrieved: {metrics["memories_retrieved"]}
💾 Memories Stored: {metrics["memories_stored"]}
⚡ Avg Latency: {avg_latency:.0f}ms
"""


@cl.set_starters
async def set_starters():
    """Set conversation starters."""
    return [
        cl.Starter(
            label="Introduce yourself",
            message="Hi! My name is Alex and I work at TechCorp as a software engineer.",
        ),
        cl.Starter(
            label="Share project details",
            message="My current project has a budget of $75,000 and the deadline is March 15th.",
        ),
        cl.Starter(
            label="Test memory recall",
            message="What do you remember about me?",
        ),
        cl.Starter(
            label="Start new session",
            message="/new",
        ),
    ]


@cl.on_chat_start
async def start():
    """Initialize the chat session."""
    # Get authenticated user
    user = cl.user_session.get("user")
    user_id = user.identifier if user else "anonymous"
    user_name = user.metadata.get("name", user_id) if user else "User"
    
    # Store in session for easy access
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("user_name", user_name)
    cl.user_session.set("previous_response_id", None)
    cl.user_session.set("turn_count", 0)  # Reset turn count for new chat
    cl.user_session.set("session_start", datetime.now().isoformat())
    
    # Simple branded header only
    header = """## 🚀 Semantic Router

*Agentic Memory Demo*"""
    
    await cl.Message(content=header).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    user_id = cl.user_session.get("user_id", "anonymous")
    user_name = cl.user_session.get("user_name", "User")
    previous_response_id = cl.user_session.get("previous_response_id")
    metrics = get_user_metrics(user_id)
    
    # Track turn count for this session (for extraction batch tracking)
    # 0-based counting to match router's extraction logic
    turn_count = cl.user_session.get("turn_count", 0)
    cl.user_session.set("turn_count", turn_count + 1)  # Increment for next turn
    
    # Handle special commands
    if message.content.lower() == "/new":
        cl.user_session.set("previous_response_id", None)
        cl.user_session.set("turn_count", 0)
        await cl.Message(
            content=f"""## 🔄 Conversation Reset

Memory persists for user **{user_name}**!

💡 *Tip: Use the ✏️ pencil icon in the sidebar for a cleaner new chat experience.*

Try asking: *"What do you remember about me?"*"""
        ).send()
        return
    
    if message.content.lower() == "/stats":
        await cl.Message(content=update_metrics_display(user_id)).send()
        return
    
    # Build request for semantic router
    request_body = {
        "model": MODEL,
        "input": message.content,
        "memory_config": {
            "enabled": True,
            "auto_store": True,
        },
        "memory_context": {
            "user_id": f"demo_{user_id}",
        }
    }
    
    # Chain to previous response if exists
    if previous_response_id:
        request_body["previous_response_id"] = previous_response_id
    
    # Create response message with step indicators
    msg = cl.Message(content="")
    await msg.send()
    
    # Track latency
    start_time = time.time()
    
    # Extraction logic: batch_size=1 means extract every 2 turns (1, 3, 5...)
    # batch_size=2 means extract every 3 turns (2, 5, 8...)
    # Turn 0 always skipped (no history)
    BATCH_SIZE = 1
    will_extract = (turn_count >= BATCH_SIZE) and ((turn_count - BATCH_SIZE) % (BATCH_SIZE + 1) == 0)
    
    try:
        # Make request to semantic router
        response = requests.post(
            f"{ROUTER_URL}/v1/responses",
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        
        # Update metrics
        latency_ms = (time.time() - start_time) * 1000
        metrics["total_requests"] += 1
        metrics["latencies"].append(latency_ms)
        output_text = data.get("output_text", "")
        
        # Step 1: Memory Retrieval indicator
        async with cl.Step(name="🔍 Memory Retrieval", type="tool") as step:
            # Check if this looks like a recall question
            recall_keywords = ["remember", "recall", "know about", "what is my", "what's my", "who am i", "where do i"]
            if any(kw in message.content.lower() for kw in recall_keywords):
                metrics["memories_retrieved"] += 1
                step.output = "✅ Retrieved memories from Milvus"
            else:
                step.output = "ℹ️ No memory retrieval needed"
        
        # Step 2: Memory Extraction indicator (only on batch turns)
        async with cl.Step(name="💾 Memory Extraction", type="tool") as step:
            if will_extract:
                metrics["memories_stored"] += 1
                step.output = f"✅ Turn {turn_count}, extracting every {BATCH_SIZE} turns - NOW EXTRACTING"
            else:
                step.output = f"⏳ Turn {turn_count}, extracting every {BATCH_SIZE} turns - not extracting"
        
        # Extract response
        response_id = data.get("id")
        conversation_id = data.get("conversation_id", "N/A")
        usage = data.get("usage", {})
        
        # Store for conversation chaining
        cl.user_session.set("previous_response_id", response_id)
        
        # Format response with metadata
        response_content = f"""{output_text}

---
<sub>⚡ {latency_ms:.0f}ms | 📝 {usage.get('total_tokens', 'N/A')} tokens | 🆔 {conversation_id[:20]}...</sub>"""
        
        msg.content = response_content
        await msg.update()
        
    except requests.exceptions.ConnectionError:
        msg.content = "❌ **Error**: Cannot connect to Semantic Router. Make sure it's running on port 8801."
        await msg.update()
    except requests.exceptions.RequestException as e:
        msg.content = f"❌ **Error**: {e}"
        await msg.update()
    except Exception as e:
        msg.content = f"❌ **Unexpected Error**: {e}"
        await msg.update()
