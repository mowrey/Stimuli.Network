// comment_generator_server.js
const http = require('http');
const fs = require('fs');
const path = require('path');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");

// --- Configuration ---
const PORT = process.env.PORT || 8080;
const GEMINI_API_KEY = process.env.GEMINI_STIMULI_KEY; // Ensure this matches your environment variable
const AI_MODEL_NAME = "gemini-1.5-flash-latest";
const COMMENT_API_ENDPOINT = "/api/generate-comment";
const PING_ENDPOINT = "/api/ping";

console.log(`Using port: ${PORT}`);
console.log(`Using AI Model: ${AI_MODEL_NAME}`);
console.log(`API Endpoint: ${COMMENT_API_ENDPOINT}`);
console.log(`Ping Endpoint: ${PING_ENDPOINT}`);

// --- Validate API Key ---
if (!GEMINI_API_KEY) {
    console.error("FATAL ERROR: GEMINI_STIMULI_KEY environment variable is not set. Server cannot start.");
    process.exit(1); // Exit if the key is missing
} else {
    console.log("GEMINI_STIMULI_KEY found (length check passed)."); // Avoid logging the key itself
}

// --- Initialize AI Client ---
let genAI, aiModel;
try {
    genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    aiModel = genAI.getGenerativeModel({ model: AI_MODEL_NAME });
    console.log(`Successfully initialized Google AI model: ${AI_MODEL_NAME}`);
} catch (error) {
    console.error("FATAL ERROR: AI client initialization failed. Server cannot start.", error);
    process.exit(1);
}

// --- Define Safety Settings ---
const generationSafetySettings = [
    { category: HarmCategory.HARM_CATEGORY_HARASSMENT,          threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,         threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,   threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
    { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,   threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE },
];
console.log("Safety settings defined:", generationSafetySettings.map(s => `${s.category}: ${s.threshold}`).join(', '));


// --- AI Comment Generation Function (Generates Multiple Comments) ---
async function generateMultipleComments(postContext) {
    // Basic validation of input context
    if (!postContext || typeof postContext !== 'string' || postContext.trim() === "") {
        console.warn("generateMultipleComments: Invalid or empty context provided.");
        return { error: "Invalid context provided" }; // Return error object
    }

    // Determine random number of comments to generate
    const minComments = 10;
    const maxComments = 25;
    const numberOfComments = Math.floor(Math.random() * (maxComments - minComments + 1)) + minComments;
    console.log(`Generating ${numberOfComments} comments for context starting with: "${postContext.substring(0, 70)}..."`);

    // --- Construct the prompt for the AI (with enhanced diversity instructions) ---
    const prompt = `Based on the following online post snippet: "${postContext}"

Generate exactly ${numberOfComments} **highly distinct and varied** comments reacting to the post. Ensure each comment offers a **unique perspective or angle** compared to the others. Comments should be short (10-25 words each), realistic, relevant, constructive, and creative.
Comments should aim to be **thought-provoking**, supportive, curious, **offer an insightful perspective,** or provide a brief related thought that **builds upon the post's idea**.
**Crucially, avoid repeating similar phrases or sentence structures across the comments.**
Do not use hashtags. Do not introduce yourself (e.g., "As an AI..."). Avoid generic questions unless they genuinely add significant value or insight.

Output ONLY a valid JSON array containing exactly ${numberOfComments} strings, where each string is one comment. Example format: ["Comment 1 text.", "Comment 2 text.", ..., "Comment ${numberOfComments} text."]`;
    // --- End of prompt modification ---

    try {
        // Make the API call to Gemini
        const result = await aiModel.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: {
                temperature: 0.9, // Keep temperature relatively high for variety
                topP: 0.95,
                maxOutputTokens: 2048 // Allow enough tokens for the JSON array
            },
            safetySettings: generationSafetySettings
        });

        const response = result.response;

        // --- Process the AI Response ---
        if (response) {
            const candidate = response.candidates?.[0]; // Get the first candidate

            // Check for safety blocks first
            if (candidate?.safetyRatings?.some(rating => rating.probability !== 'NEGLIGIBLE' && rating.probability !== 'LOW')) {
                 console.warn(`AI Comment Gen Blocked by Safety Filter:`, candidate.safetyRatings);
                 return { error: "AI response blocked by safety filter" };
            }
            // Check for prompt blocks
            const blockReason = response.promptFeedback?.blockReason;
            if (blockReason) {
                console.error(`AI Comment Gen Blocked by Prompt Filter: Reason: ${blockReason}`);
                return { error: `AI response blocked: ${blockReason}` };
            }
            // Check for other finish reasons
            if (candidate?.finishReason && candidate.finishReason !== "STOP" && candidate.finishReason !== "MAX_TOKENS") {
                console.warn(`AI Comment Gen Stopped Early: Reason: ${candidate.finishReason}`);
            }

            // Attempt to extract and parse the text content
            if (candidate?.content?.parts?.[0]?.text) {
                const rawText = candidate.content.parts[0].text.trim();
                let jsonString = rawText;

                // Clean potential Markdown fences
                const jsonRegex = /^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$/;
                const match = rawText.match(jsonRegex);
                if (match && match[1]) {
                    jsonString = match[1].trim();
                }

                // Attempt to parse the potentially cleaned JSON string
                try {
                    let parsedComments = JSON.parse(jsonString);
                    // Validate if it's an array of strings
                    if (Array.isArray(parsedComments) && parsedComments.every(item => typeof item === 'string')) {
                        console.log(`AI Comment Gen Success: Received ${parsedComments.length} comments.`);
                        return { comments: parsedComments }; // Success: Return the array
                    } else {
                        console.warn("AI Gen Comment: Parsed JSON is not an array of strings.", jsonString);
                        return { error: "AI response format incorrect (not an array of strings)." };
                    }
                } catch (parseError) {
                    console.error("AI Gen Comment: Failed to parse AI response as JSON.", parseError, "Attempted to parse:", jsonString);
                    return { error: "AI response format incorrect (failed to parse JSON)." };
                }
            } else {
                console.warn("AI Gen Comment: No valid text content part found in candidate.");
                return { error: "AI response format unexpected (no text part)." };
            }
        } else {
            console.error("AI Gen Comment: No response object found in the result.");
            return { error: "AI generation failed: No response received." };
        }

    } catch (error) {
        console.error(`AI Comment Gen API error during call:`, error);
        const errorMessage = error.message || "Unknown error";
        return { error: `AI generation failed: ${errorMessage.substring(0, 100)}` }; // Return truncated error
    }
}


// --- Create HTTP Server ---
console.log("Creating HTTP server...");
const server = http.createServer(async (req, res) => {
    const requestTimestamp = new Date().toISOString();
    const logPrefix = `[${requestTimestamp} ${req.method} ${req.url}]`;
    console.log(`\n${logPrefix} Request received.`);

    // --- CORS Headers ---
    res.setHeader('Access-Control-Allow-Origin', '*'); // Consider restricting in production
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS, GET');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    // --- Handle OPTIONS Preflight ---
    if (req.method === 'OPTIONS') {
        console.log(`${logPrefix} Handling OPTIONS preflight.`);
        res.writeHead(204);
        res.end();
        console.log(`${logPrefix} Responded 204.`);
        return;
    }

    // --- Handle POST /api/generate-comment ---
    if (req.method === 'POST' && req.url === COMMENT_API_ENDPOINT) {
        console.log(`${logPrefix} Handling API request.`);
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('error', (err) => {
             console.error(`${logPrefix} Request body stream error:`, err);
             if (!res.headersSent) {
                 res.writeHead(400, { 'Content-Type': 'application/json' });
                 res.end(JSON.stringify({ error: 'Error reading request body.' }));
             }
        });
        req.on('end', async () => {
            console.log(`${logPrefix} Request body received (${body.length} bytes).`);
            try {
                if (!body) {
                    console.warn(`${logPrefix} Empty request body.`);
                    res.writeHead(400, {'Content-Type': 'application/json'});
                    res.end(JSON.stringify({ error: 'Empty request body.'}));
                    console.log(`${logPrefix} Responded 400 (Empty Body).`);
                    return;
                 }
                let requestData;
                try {
                    requestData = JSON.parse(body);
                } catch (parseError) {
                    console.error(`${logPrefix} Invalid JSON in request body:`, parseError);
                    res.writeHead(400, {'Content-Type': 'application/json'});
                    res.end(JSON.stringify({ error: 'Invalid JSON format in request body.' }));
                    console.log(`${logPrefix} Responded 400 (Invalid JSON).`);
                    return;
                }
                const postContext = requestData.context;
                if (!postContext || typeof postContext !== 'string') {
                    console.warn(`${logPrefix} Invalid or missing 'context' in request body:`, requestData);
                    res.writeHead(400, {'Content-Type': 'application/json'});
                    res.end(JSON.stringify({ error: "Request body must contain a 'context' field as a string." }));
                    console.log(`${logPrefix} Responded 400 (Missing Context).`);
                    return;
                }

                // Generate multiple comments
                const generationResult = await generateMultipleComments(postContext);

                // Handle potential errors from generation
                if (generationResult.error) {
                    console.error(`${logPrefix} AI generation failed: ${generationResult.error}`);
                    const statusCode = (generationResult.error.includes("blocked by safety") || generationResult.error.includes("blocked by Prompt") || generationResult.error.includes("format incorrect")) ? 400 : 500;
                    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: `Failed to generate comments: ${generationResult.error}` }));
                    console.log(`${logPrefix} Responded ${statusCode} due to generation error.`);
                } else {
                    // Send successful array of comments
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ comments: generationResult.comments })); // Send the array under the "comments" key
                    console.log(`${logPrefix} Responded 200 with ${generationResult.comments?.length || 0} comments.`);
                }

            } catch (error) { // Catch unexpected errors in the endpoint handler
                console.error(`${logPrefix} Unexpected error processing API request:`, error);
                if (!res.headersSent) {
                     res.writeHead(500, { 'Content-Type': 'application/json' });
                     res.end(JSON.stringify({ error: 'Internal server error processing request.' }));
                     console.log(`${logPrefix} Responded 500 (Handler Error).`);
                } else {
                     console.error(`${logPrefix} Headers already sent, could not send 500 response.`);
                }
            }
        });
    }
    // --- Handle GET /api/ping ---
    else if (req.method === 'GET' && req.url === PING_ENDPOINT) {
        console.log(`${logPrefix} Handling PING request.`);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ status: 'awake' }));
        console.log(`${logPrefix} Responded 200 (Ping OK).`);
    }
    // --- Handle GET / (Serve index.html) ---
    else if (req.method === 'GET' && req.url === '/') {
        console.log(`${logPrefix} Handling GET request for root path /`);
        const filePath = path.join(__dirname, 'index.html'); // Assumes index.html is in the same directory
        console.log(`${logPrefix} Attempting to read file: ${filePath}`);

        fs.readFile(filePath, 'utf8', (err, content) => {
            if (err) {
                console.error(`${logPrefix} Error reading index.html file:`, err);
                if (err.code === 'ENOENT') {
                    res.writeHead(404, { 'Content-Type': 'text/plain' });
                    res.end('404 Not Found: index.html missing.');
                    console.log(`${logPrefix} Responded 404 (File Not Found).`);
                } else {
                    res.writeHead(500, { 'Content-Type': 'text/plain' });
                    res.end('500 Internal Server Error.');
                    console.log(`${logPrefix} Responded 500 (File Read Error).`);
                }
            } else {
                console.log(`${logPrefix} Successfully read index.html.`);
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(content);
                console.log(`${logPrefix} Responded 200 with index.html.`);
            }
        });
    }
    // --- Handle all other requests ---
    else {
        if (req.url !== '/favicon.ico') { // Ignore favicon requests for cleaner logs
            console.log(`${logPrefix} Responding 404 Not Found.`);
        }
        res.writeHead(404, { 'Content-Type': 'text/plain' });
        res.end('Not Found');
    }
});
console.log("HTTP server created.");

// --- Start the HTTP Server ---
console.log(`Attempting to start server listening on port ${PORT}...`);
server.listen(PORT, () => {
    console.log(`-------------------------------------------------------`);
    console.log(`  HTTP Server for Comment Generation is RUNNING!`);
    console.log(`  Listening on port: ${PORT}`);
    console.log(`  Serving index.html at: GET /`);
    console.log(`  API endpoint available at: POST ${COMMENT_API_ENDPOINT}`);
    console.log(`  Ping endpoint available at: GET ${PING_ENDPOINT}`);
    console.log(`-------------------------------------------------------`);
});

// --- Handle Server Start Errors ---
server.on('error', (error) => {
    console.error("Server failed to start:", error);
    if (error.code === 'EADDRINUSE') {
        console.error(`Port ${PORT} is already in use. Is another server running?`);
    }
    process.exit(1); // Exit if server cannot start
});

// --- Graceful Shutdown Logic ---
function shutdown() {
  console.log('\nReceived kill signal, shutting down gracefully...');
  server.close(() => {
    console.log('Closed out remaining connections.');
    process.exit(0); // Exit successfully after closing
  });

  // Force shutdown if server hasn't closed within a timeout
  setTimeout(() => {
    console.error('Could not close connections in time, forcefully shutting down');
    process.exit(1); // Exit with error code
  }, 10000); // 10 seconds timeout
}

process.on('SIGTERM', shutdown); // Standard signal for termination (e.g., from Render)
process.on('SIGINT', shutdown);  // Signal for Ctrl+C in terminal

console.log("Server setup complete. Waiting for requests...");
