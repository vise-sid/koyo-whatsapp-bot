# Firebase Setup Guide

## Environment Variables

Add these environment variables to your deployment:

### Option 1: Service Account Key (Recommended for Production)
```bash
FIREBASE_CREDENTIALS='{"type": "service_account", "project_id": "your-project-id", "private_key_id": "...", "private_key": "...", "client_email": "...", "client_id": "...", "auth_uri": "...", "token_uri": "...", "auth_provider_x509_cert_url": "...", "client_x509_cert_url": "..."}'
```

### Option 2: Default Credentials (For Local Development)
- Set up Firebase CLI: `firebase login`
- Or use a service account key file: `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json`

## Firebase Console Setup

1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project or select existing one
3. Go to Project Settings > Service Accounts
4. Generate new private key
5. Copy the JSON content and use it as `FIREBASE_CREDENTIALS` environment variable

## Firestore Database Structure

The app uses a nested collection structure for better organization:

### Collection Path
```
users/{user_id}/conversations/{character_name}/messages/{message_id}
```

### Example Structure
```
users/
├── +919821785549/                  # User document (phone number from WhatsApp/Twilio)
│   └── conversations/
│       └── meher/                  # Character name (always "meher")
│           ├── last_updated: "2025-01-24T10:30:00Z"
│           ├── message_count: 23
│           ├── character_name: "meher"
│           ├── call_sid: "CAf5f9ad612e5807d77d513bdd2252332a" // Only for voice
│           └── messages/
│               ├── {message_id_1}/
│               │   ├── sender: "user"
│               │   ├── content: "Hello Meher!"
│               │   ├── timestamp: "2025-01-24T10:30:00Z"
│               │   ├── sync: false
│               │   ├── conversation_type: "text"
│               │   └── call_sid: null
│               ├── {message_id_2}/
│               │   ├── sender: "character"
│               │   ├── content: "Hi there! How are you doing?"
│               │   ├── timestamp: "2025-01-24T10:30:15Z"
│               │   ├── sync: false
│               │   ├── conversation_type: "text"
│               │   └── call_sid: null
│               └── {message_id_3}/
│                   ├── sender: "user"
│                   ├── content: "What's the weather like?"
│                   ├── timestamp: "2025-01-24T10:30:00Z"
│                   ├── sync: false
│                   ├── conversation_type: "voice"
│                   └── call_sid: "CAf5f9ad612e5807d77d513bdd2252332a"
```

### Structure Details
- **user_id**: Phone number extracted from WhatsApp/Twilio (e.g., "+919821785549")
- **character_name**: Always "meher" (the AI character)
- **All conversations** (both text and voice) are stored under the same character document
- **Message separation**: Text and voice messages are distinguished by the `conversation_type` field

### Conversation Metadata Fields
- **last_updated**: When the conversation was last updated
- **message_count**: Total number of messages in the conversation
- **character_name**: "meher" (always the same)
- **call_sid**: Twilio Call SID (only present for voice conversations)

### Message Fields
- **sender**: "user" or "character" (Meher)
- **content**: Message content (limited to 1000 characters)
- **timestamp**: When the message was sent
- **sync**: Sync status (defaults to false)
- **conversation_type**: "text" or "voice"
- **call_sid**: Twilio Call SID (only for voice conversations, null for text)

## Testing

1. Start your app
2. Send a WhatsApp message or make a voice call
3. Check Firestore console to see messages being saved
4. Look for logs: "Saved [sender] message to Firebase for user [user_id]"
