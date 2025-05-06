// comment_generator_server.js
const http = require('http');
const fs = require('fs');
const path = require('path');
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");

// --- Configuration ---
const PORT = process.env.PORT || 8080;
const GEMINI_API_KEY = process.env.GEMINI_STIMULI_KEY;
const AI_MODEL_NAME = "gemini-1.5-flash-latest";
const COMMENT_API_ENDPOINT = "/api/generate-comment";
const POST_CONTENT_API_ENDPOINT = "/api/generate-post-content"; // <<< NEW ENDPOINT
const PING_ENDPOINT = "/api/ping";

console.log(`Using port: ${PORT}`);
console.log(`Using AI Model: ${AI_MODEL_NAME}`);
console.log(`Comment API Endpoint: ${COMMENT_API_ENDPOINT}`);
console.log(`Post Content API Endpoint: ${POST_CONTENT_API_ENDPOINT}`); // <<< LOG NEW
console.log(`Ping Endpoint: ${PING_ENDPOINT}`);

// --- Validate API Key ---
if (!GEMINI_API_KEY) {
    console.error("FATAL ERROR: GEMINI_STIMULI_KEY environment variable is not set. Server cannot start.");
    process.exit(1);
} else {
    console.log("GEMINI_STIMULI_KEY found (length check passed).");
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


// --- AI Function for generating POST CONTENT ---
async function generateSinglePostContent(themeText) {
    if (!themeText || typeof themeText !== 'string' || themeText.trim() === "") {
        console.warn("generateSinglePostContent: Invalid or empty theme provided.");
        return { error: "Invalid theme provided for post content generation" };
    }
    console.log(`Generating post content based on theme: "${themeText.substring(0, 70)}..."`);

    const prompt = `Generate a community update post of about 250-350 characters, consisting of 2-3 paragraphs, expanding on the theme: "${themeText}".
Focus on constructive engagement, community building, or upcoming initiatives.
The output should be plain text, with paragraphs separated by a double newline (\\n\\n).
Do not include a title or any preambles like "Here's a post:". Just the post content.`;

    try {
        const result = await aiModel.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: {
                temperature: 0.7, // Slightly lower temp for more focused post content
                topP: 0.95,
                maxOutputTokens: 512 // Enough for a few paragraphs
            },
            safetySettings: generationSafetySettings
        });
        const response = result.response;
        if (response) {
            const candidate = response.candidates?.[0];
            if (candidate?.safetyRatings?.some(rating => rating.probability !== 'NEGLIGIBLE' && rating.probability !== 'LOW')) {
                 console.warn(`AI Post Content Gen Blocked by Safety Filter:`, candidate.safetyRatings);
                 return { error: "AI response blocked by safety filter" };
            }
            const blockReason = response.promptFeedback?.blockReason;
            if (blockReason) {
                console.error(`AI Post Content Gen Blocked by Prompt Filter: Reason: ${blockReason}`);
                return { error: `AI response blocked: ${blockReason}` };
            }
            if (candidate?.content?.parts?.[0]?.text) {
                const postText = candidate.content.parts[0].text.trim();
                console.log(`AI Post Content Gen Success. Length: ${postText.length}`);
                return { postText: postText }; // Success
            } else {
                console.warn("AI Post Content Gen: No valid text content part found.");
                return { error: "AI response format unexpected (no text part)." };
            }
        } else {
            console.error("AI Post Content Gen: No response object found.");
            return { error: "AI generation failed: No response received." };
        }
    } catch (error) {
        console.error(`AI Post Content Gen API error:`, error);
        return { error: `AI generation failed: ${error.message?.substring(0, 100) || "Unknown error"}` };
    }
}


// --- AI Comment Generation Function (Generates Multiple Comments for a post's comment section) ---
async function generateMultipleComments(postContext) {
    if (!postContext || typeof postContext !== 'string' || postContext.trim() === "") {
        console.warn("generateMultipleComments: Invalid or empty context provided.");
        return { error: "Invalid context provided" };
    }
    const minComments = 10;
    const maxComments = 25;
    const numberOfComments = Math.floor(Math.random() * (maxComments - minComments + 1)) + minComments;
    console.log(`Generating ${numberOfComments} comments for context starting with: "${postContext.substring(0, 70)}..."`);

    const prompt = `Based on the following online post snippet: "${postContext}"

Generate exactly ${numberOfComments} **highly distinct and varied** comments reacting to the post. Ensure each comment offers a **unique perspective or angle** compared to the others. Comments should be short (10-25 words each), realistic, relevant, constructive, and creative.
Comments should aim to be **thought-provoking**, supportive, curious, **offer an insightful perspective,** or provide a brief related thought that **builds upon the post's idea**.
**Crucially, avoid repeating similar phrases or sentence structures across the comments.**
Do not use hashtags. Do not introduce yourself (e.g., "As an AI..."). Avoid generic questions unless they genuinely add significant value or insight.

Output ONLY a valid JSON array containing exactly ${numberOfComments} strings, where each string is one comment. Example format: ["Comment 1 text.", "Comment 2 text.", ..., "Comment ${numberOfComments} text."]`;

    try {
        const result = await aiModel.generateContent({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: {
                temperature: 0.9,
                topP: 0.95,
                maxOutputTokens: 2048
            },
            safetySettings: generationSafetySettings
        });
        const response = result.response;
        if (response) {
            const candidate = response.candidates?.[0];
            if (candidate?.safetyRatings?.some(rating => rating.probability !== 'NEGLIGIBLE' && rating.probability !== 'LOW')) {
                 console.warn(`AI Comment Gen Blocked by Safety Filter:`, candidate.safetyRatings);
                 return { error: "AI response blocked by safety filter" };
            }
            const blockReason = response.promptFeedback?.blockReason;
            if (blockReason) {
                console.error(`AI Comment Gen Blocked by Prompt Filter: Reason: ${blockReason}`);
                return { error: `AI response blocked: ${blockReason}` };
            }
            if (candidate?.finishReason && candidate.finishReason !== "STOP" && candidate.finishReason !== "MAX_TOKENS") {
                console.warn(`AI Comment Gen Stopped Early: Reason: ${candidate.finishReason}`);
            }
            if (candidate?.content?.parts?.[0]?.text) {
                const rawText = candidate.content.parts[0].text.trim();
                let jsonString = rawText;
                const jsonRegex = /^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$/;
                const match = rawText.match(jsonRegex);
                if (match && match[1]) {
                    jsonString = match[1].trim();
                }
                try {
                    let parsedComments = JSON.parse(jsonString);
                    if (Array.isArray(parsedComments) && parsedComments.every(item => typeof item === 'string')) {
                        console.log(`AI Comment Gen Success: Received ${parsedComments.length} comments.`);
                        return { comments: parsedComments };
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
        return { error: `AI generation failed: ${errorMessage.substring(0, 100)}` };
    }
}


// --- Create HTTP Server ---
console.log("Creating HTTP server...");
const server = http.createServer(async (req, res) => {
    const requestTimestamp = new Date().toISOString();
    const logPrefix = `[${requestTimestamp} ${req.method} ${req.url}]`;
    console.log(`\n${logPrefix} Request received.`);

    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS, GET');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

    if (req.method === 'OPTIONS') {
        console.log(`${logPrefix} Handling OPTIONS preflight.`);
        res.writeHead(204);
        res.end();
        console.log(`${logPrefix} Responded 204.`);
        return;
    }

    // --- ROUTING ---

    // Endpoint for generating a BATCH OF COMMENTS for a comment section
    if (req.method === 'POST' && req.url === COMMENT_API_ENDPOINT) {
        console.log(`${logPrefix} Handling /api/generate-comment request.`);
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('error', (err) => { /* ... error handling ... */ });
        req.on('end', async () => {
            try {
                if (!body) { /* ... empty body error ... */ return; }
                let requestData;
                try { requestData = JSON.parse(body); }
                catch (parseError) { /* ... invalid JSON error ... */ return; }
                const postContext = requestData.context; // Context for generating comments *about* this
                if (!postContext || typeof postContext !== 'string') { /* ... missing context error ... */ return; }

                const generationResult = await generateMultipleComments(postContext); // Call the correct function

                if (generationResult.error) {
                    console.error(`${logPrefix} Multiple comments generation failed: ${generationResult.error}`);
                    const statusCode = (generationResult.error.includes("blocked") || generationResult.error.includes("format incorrect")) ? 400 : 500;
                    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: `Failed to generate comments: ${generationResult.error}` }));
                } else {
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ comments: generationResult.comments }));
                    console.log(`${logPrefix} Responded 200 with ${generationResult.comments?.length || 0} comments.`);
                }
            } catch (error) { /* ... unexpected handler error ... */ }
        });
    }
    // Endpoint for generating SINGLE POST CONTENT based on a theme
    else if (req.method === 'POST' && req.url === POST_CONTENT_API_ENDPOINT) {
        console.log(`${logPrefix} Handling /api/generate-post-content request.`);
        let body = '';
        req.on('data', chunk => { body += chunk.toString(); });
        req.on('error', (err) => { /* ... error handling ... */ });
        req.on('end', async () => {
            try {
                if (!body) { /* ... empty body error ... */ return; }
                let requestData;
                try { requestData = JSON.parse(body); }
                catch (parseError) { /* ... invalid JSON error ... */ return; }
                const themeText = requestData.theme; // Expecting a "theme" or "subheadline" for post content
                if (!themeText || typeof themeText !== 'string') {
                    console.warn(`${logPrefix} Invalid or missing 'theme' for post content:`, requestData);
                    res.writeHead(400, {'Content-Type': 'application/json'});
                    res.end(JSON.stringify({ error: "Request body must contain a 'theme' field as a string." }));
                    return;
                }

                const generationResult = await generateSinglePostContent(themeText); // Call the new function

                if (generationResult.error) {
                    console.error(`${logPrefix} Single post content generation failed: ${generationResult.error}`);
                    const statusCode = (generationResult.error.includes("blocked")) ? 400 : 500;
                    res.writeHead(statusCode, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ error: `Failed to generate post content: ${generationResult.error}` }));
                } else {
                    res.writeHead(200, { 'Content-Type': 'application/json' });
                    res.end(JSON.stringify({ postText: generationResult.postText })); // Send single text under "postText"
                    console.log(`${logPrefix} Responded 200 with generated post content.`);
                }
            } catch (error) { /* ... unexpected handler error ... */ }
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
        const filePath = path.join(__dirname, 'index.html');
        console.log(`${logPrefix} Attempting to read file: ${filePath}`);

        fs.readFile(filePath, 'utf8', (err, content) => {
            if (err) {
                console.error(`${logPrefix} Error reading index.html file:`, err);
                if (err.code === 'ENOENT') {
                    res.writeHead(404, { 'Content-Type': 'text/plain' });
                    res.end('404 Not Found: index.html missing.');
                } else {
                    res.writeHead(500, { 'Content-Type': 'text/plain' });
                    res.end('500 Internal Server Error.');
                }
            } else {
                res.writeHead(200, { 'Content-Type': 'text/html' });
                res.end(content);
            }
        });
    }
    // --- Handle all other requests ---
    else {
        if (req.url !== '/favicon.ico') {
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
    console.log(`  HTTP Server is RUNNING!`);
    console.log(`  Listening on port: ${PORT}`);
    console.log(`  Serving index.html at: GET /`);
    console.log(`  Comment API: POST ${COMMENT_API_ENDPOINT}`);
    console.log(`  Post Content API: POST ${POST_CONTENT_API_ENDPOINT}`);
    console.log(`  Ping API: GET ${PING_ENDPOINT}`);
    console.log(`-------------------------------------------------------`);
});

// --- Handle Server Start Errors ---
server.on('error', (error) => {
    console.error("Server failed to start:", error);
    if (error.code === 'EADDRINUSE') {
        console.error(`Port ${PORT} is already in use. Is another server running?`);
    }
    process.exit(1);
});

// --- Graceful Shutdown Logic ---
function shutdown() {
  console.log('\nReceived kill signal, shutting down gracefully...');
  server.close(() => {
    console.log('Closed out remaining connections.');
    process.exit(0);
  });
  setTimeout(() => {
    console.error('Could not close connections in time, forcefully shutting down');
    process.exit(1);
  }, 10000);
}
process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

console.log("Server setup complete. Waiting for requests...");
