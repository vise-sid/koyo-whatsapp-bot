/**
 * Firebase Cloud Functions for batch memory processing
 * 
 * This function triggers when a new message is added to Firebase
 * and checks if batch processing is needed for that user-character pair.
 */

const functions = require('firebase-functions');
const admin = require('firebase-admin');
const { spawn } = require('child_process');

// Initialize Firebase Admin
admin.initializeApp();

/**
 * Cloud Function that triggers on new message creation
 * Checks if batch processing is needed and triggers it if so
 */
exports.checkBatchTrigger = functions.firestore
  .document('users/{userId}/conversations/{characterName}/messages/{messageId}')
  .onCreate(async (snap, context) => {
    const { userId, characterName } = context.params;
    const messageData = snap.data();
    
    console.log(`New message created for user ${userId} and character ${characterName}`);
    
    try {
      // Count unsynced messages for this user-character pair
      const unsyncedCount = await countUnsyncedMessages(userId, characterName);
      
      console.log(`Found ${unsyncedCount} unsynced messages for user ${userId}`);
      
      // If we have enough unsynced messages, trigger batch processing
      if (unsyncedCount >= 25) {
        console.log(`Triggering batch processing for user ${userId} (${unsyncedCount} unsynced messages)`);
        
        // Trigger batch processing
        await triggerBatchProcessing(userId, characterName);
      }
      
    } catch (error) {
      console.error(`Error in batch trigger for user ${userId}:`, error);
    }
  });

/**
 * Count unsynced messages for a user-character pair
 */
async function countUnsyncedMessages(userId, characterName) {
  try {
    const messagesRef = admin.firestore()
      .collection('users')
      .doc(userId)
      .collection('conversations')
      .doc(characterName)
      .collection('messages');
    
    const unsyncedQuery = messagesRef.where('sync', '==', false);
    const unsyncedDocs = await unsyncedQuery.get();
    
    return unsyncedDocs.size;
  } catch (error) {
    console.error('Error counting unsynced messages:', error);
    return 0;
  }
}

/**
 * Trigger batch processing for a user-character pair
 */
async function triggerBatchProcessing(userId, characterName) {
  try {
    // Call your Python batch processing service
    // This could be an HTTP endpoint or a direct function call
    const batchProcessor = spawn('python3', [
      '-c',
      `
import asyncio
import sys
import os
sys.path.append('${__dirname}/..')

from services.batch_memory_processor import BatchMemoryProcessor

async def process_batch():
    processor = BatchMemoryProcessor('${process.env.OPENAI_API_KEY}')
    await processor.initialize()
    success = await processor.process_user_character_batch('${userId}', '${characterName}')
    print(f"Batch processing result: {success}")

asyncio.run(process_batch())
      `
    ]);
    
    batchProcessor.stdout.on('data', (data) => {
      console.log(`Batch processor output: ${data}`);
    });
    
    batchProcessor.stderr.on('data', (data) => {
      console.error(`Batch processor error: ${data}`);
    });
    
    batchProcessor.on('close', (code) => {
      console.log(`Batch processor exited with code ${code}`);
    });
    
  } catch (error) {
    console.error('Error triggering batch processing:', error);
  }
}

/**
 * Alternative: HTTP endpoint for manual batch processing
 */
exports.processBatch = functions.https.onRequest(async (req, res) => {
  try {
    const { userId, characterName } = req.body;
    
    if (!userId || !characterName) {
      return res.status(400).json({ error: 'userId and characterName are required' });
    }
    
    console.log(`Manual batch processing requested for user ${userId} and character ${characterName}`);
    
    // Trigger batch processing
    await triggerBatchProcessing(userId, characterName);
    
    res.json({ 
      success: true, 
      message: `Batch processing triggered for user ${userId}` 
    });
    
  } catch (error) {
    console.error('Error in manual batch processing:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});

/**
 * Health check endpoint
 */
exports.healthCheck = functions.https.onRequest(async (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: new Date().toISOString(),
    service: 'batch-memory-processor'
  });
});
