# Memory Implementation Guide

## Overview

This implementation provides a real-time batch processing memory system that:
- Uses Firebase as the primary storage (source of truth)
- Automatically triggers batch processing when 25+ unsynced messages accumulate
- Extracts conversation-level memories using OpenAI
- Stores semantic memories in Qdrant for search
- Provides context-aware responses for both voice and WhatsApp

## Architecture

```
Firebase (Primary) → Batch Processor → Qdrant (Semantic Search)
     ↓                    ↓                    ↓
Individual Messages → Conversation Memory → Context for AI
```

## Components

### 1. Batch Memory Processor (`services/batch_memory_processor.py`)
- Processes batches of 25+ messages
- Extracts conversation-level memories using OpenAI
- Stores memories in Qdrant with semantic search
- Marks Firebase messages as synced

### 2. Memory Integration (`services/memory_integration.py`)
- Easy integration with existing services
- Provides conversation context for AI responses
- Handles memory retrieval and formatting

### 3. Firebase Integration
- Automatic trigger when messages are saved
- Checks unsynced message count
- Triggers batch processing when threshold is reached

### 4. API Endpoints
- `/memory/check-batch/{user_id}/{character_name}` - Check if batch processing is needed
- `/memory/search/{user_id}/{character_name}` - Search for relevant memories
- `/firebase/message-created` - Firebase trigger endpoint

## Usage

### For Voice Services

```python
from services.memory_integration import memory_integration

# In your voice session handler
async def handle_voice_message(user_id: str, message: str):
    # Get conversation context
    context = await memory_integration.get_conversation_context(
        user_id=user_id,
        character_name="meher",
        current_message=message
    )
    
    # Enhance your system prompt
    enhanced_prompt = await memory_integration.enhance_system_prompt(
        user_id=user_id,
        character_name="meher",
        current_message=message,
        base_system_prompt="You are Meher, a helpful AI assistant."
    )
    
    # Use enhanced prompt for AI response
    response = await generate_ai_response(enhanced_prompt, message)
    return response
```

### For WhatsApp Services

```python
from services.memory_integration import memory_integration

# In your WhatsApp handler
async def handle_whatsapp_message(user_id: str, message: str):
    # Get conversation context
    context = await memory_integration.get_conversation_context(
        user_id=user_id,
        character_name="meher",
        current_message=message
    )
    
    # Generate response with context
    response = await generate_ai_response_with_context(message, context)
    return response
```

## Environment Variables

Add these to your `.env` file:

```bash
# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=chat_memory
QDRANT_VECTOR_SIZE=1536
QDRANT_CONNECTION_TIMEOUT=30

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
```

## API Endpoints

### Check Batch Processing
```bash
POST /memory/check-batch/{user_id}/{character_name}
```

### Search Memories
```bash
GET /memory/search/{user_id}/{character_name}?query=your_search_query&limit=5
```

### Firebase Trigger
```bash
POST /firebase/message-created
{
    "user_id": "user123",
    "character_name": "meher"
}
```

## How It Works

1. **Message Saved**: When a message is saved to Firebase, the `sync` field is set to `false`
2. **Trigger Check**: Firebase service automatically checks unsynced message count
3. **Batch Processing**: When count reaches 25+, batch processing is triggered
4. **Memory Extraction**: OpenAI analyzes the conversation and extracts key memories
5. **Qdrant Storage**: Memories are stored in Qdrant with semantic search capabilities
6. **Sync Update**: Firebase messages are marked as `sync: true`
7. **Context Retrieval**: Future messages can retrieve relevant memories for context

## Benefits

- **Automatic**: No manual intervention required
- **Efficient**: Only processes when meaningful batch size is reached
- **Context-Aware**: AI responses are informed by conversation history
- **Scalable**: Handles any number of users and conversations
- **Cost-Effective**: Minimal infrastructure requirements

## Monitoring

Check memory statistics:
```bash
GET /memory/stats/{user_id}/{character_name}
```

Health check:
```bash
GET /memory/health
```

## Deployment

1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables
3. Start your FastAPI application
4. The memory system will automatically initialize and start processing

## Troubleshooting

- Check logs for batch processing status
- Monitor unsynced message counts
- Verify Qdrant connection
- Check OpenAI API key and limits

The system is designed to be non-intrusive and will continue working even if memory processing fails, ensuring your core functionality remains unaffected.
