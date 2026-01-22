import fs from "fs";
import path from "path";
import express from "express";
import OpenAI from "openai";
const MIDDLEWARE_TOKEN = process.env.MIDDLEWARE_TOKEN;

const app = express();
app.use(express.json({ limit: "1mb" }));

// ---- Config ----
const PORT = process.env.PORT || 3000;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) throw new Error("Missing OPENAI_API_KEY");

const MODEL = process.env.OPENAI_MODEL || "gpt-5-mini"; // אפשר לשנות בלי קוד
const EMBEDDING_MODEL = process.env.OPENAI_EMBEDDING_MODEL || "text-embedding-3-small";

const TOP_K_SEARCH = Number(process.env.TOP_K_SEARCH || 12);
const TOP_K_FINAL = Number(process.env.TOP_K_FINAL || 8);
const MIN_SCORE_STRONG = Number(process.env.MIN_SCORE_STRONG || 0.78);
const MIN_SCORE_WEAK = Number(process.env.MIN_SCORE_WEAK || 0.72);

const client = new OpenAI({ apiKey: OPENAI_API_KEY });

// ---- Load KB chunks ----
const kbPath = path.resolve("./kb_chunks.json");
const systemPromptPath = path.resolve("./system_prompt.txt");

const KB = JSON.parse(fs.readFileSync(kbPath, "utf8")); // [{chunk_id, text, metadata}]
const SYSTEM_PROMPT = fs.readFileSync(systemPromptPath, "utf8");

// Optional: cache embeddings on disk so you don't recompute every boot
const cachePath = path.resolve("./embeddings_cache.json");
let EMB_CACHE = {};
if (fs.existsSync(cachePath)) {
  EMB_CACHE = JSON.parse(fs.readFileSync(cachePath, "utf8"));
}

// ---- Forced seeds (Method B) ----
const FORCED_SEEDS = {
  identify_dominant_emotion_from_event: ["CK-16","CK-17","CK-18","CK-20","CK-09","CK-32"],
  offer_emotion_options_without_diagnosis: ["CK-31","CK-32","CK-16","CK-17","CK-20","CK-19"],
  anchor_choice_or_offer_minimal_balanced_action: ["CK-21","CK-23","CK-09","CK-04","CK-06","CK-12"]
};

// ---- Helpers ----
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a) {
  return Math.sqrt(dot(a, a));
}
function cosineSim(a, b) {
  const na = norm(a), nb = norm(b);
  if (na === 0 || nb === 0) return 0;
  return dot(a, b) / (na * nb);
}

async function embedText(text) {
  const key = `${EMBEDDING_MODEL}::${text}`;
  if (EMB_CACHE[key]) return EMB_CACHE[key];

  const resp = await client.embeddings.create({
    model: EMBEDDING_MODEL,
    input: text
  });
  const vec = resp.data?.[0]?.embedding;
  if (!vec) throw new Error("Embedding missing in response");

  EMB_CACHE[key] = vec;
  // Persist cache to disk (MVP). In prod, store in DB.
  fs.writeFileSync(cachePath, JSON.stringify(EMB_CACHE));
  return vec;
}

async function ensureChunkEmbeddings() {
  // Precompute embeddings for chunks if not present in cache
  for (const ch of KB) {
    const key = `${EMBEDDING_MODEL}::chunk::${ch.chunk_id}`;
    if (EMB_CACHE[key]) continue;
    const resp = await client.embeddings.create({ model: EMBEDDING_MODEL, input: ch.text });
    const vec = resp.data?.[0]?.embedding;
    if (!vec) throw new Error(`Embedding missing for chunk ${ch.chunk_id}`);
    EMB_CACHE[key] = vec;
    fs.writeFileSync(cachePath, JSON.stringify(EMB_CACHE));
  }
}

function getChunkVec(chunkId) {
  const key = `${EMBEDDING_MODEL}::chunk::${chunkId}`;
  return EMB_CACHE[key];
}

function filterByExercise(ch, exerciseId) {
  const ex = ch.metadata?.exercise_ids || [];
  return ex.includes(exerciseId);
}

function pickForcedSeeds(intent, exerciseId) {
  const ids = FORCED_SEEDS[intent] || [];
  const seeds = ids
    .map((id) => KB.find((c) => c.chunk_id === id))
    .filter(Boolean)
    .filter((c) => filterByExercise(c, exerciseId));
  return seeds;
}

function buildQueryText({ chat_intent, exercise_id, step_number, user_message }) {
  return `${chat_intent} | ${exercise_id} | step=${step_number} | ${user_message}`;
}

function buildKbContext(chunks) {
  // Internal: include IDs so the model can be constrained; do NOT show to user
  return chunks
    .map((c) => `[${c.chunk_id}] ${c.text}`)
    .join("\n");
}

function classifyRetrievalStrength(top1Score) {
  if (top1Score >= MIN_SCORE_STRONG) return "strong";
  if (top1Score >= MIN_SCORE_WEAK) return "weak";
  return "none";
}
// ---- ה-Endpoint החדש והמשופר ----
app.post("/tango/ai/coach_message", async (req, res) => {
  try {
    // אימות מול באבל
    if (MIDDLEWARE_TOKEN) {
      const auth = req.headers["authorization"] || "";
      const ok = auth === `Bearer ${MIDDLEWARE_TOKEN}` || auth === MIDDLEWARE_TOKEN;
      if (!ok) return res.status(401).json({ error: "Unauthorized" });
    }

    const {
      exercise_id,
      chat_intent,
      user_message,
      chat_history // רשימת הודעות קודמות מבאבל
    } = req.body || {};

    // בדיקה בסיסית
    if (!exercise_id || !user_message) {
      return res.status(400).json({ error: "Missing required fields" });
    }

    // 1. חיפוש מידע ב-KB (החומר התיאורטי שלך)
    // אנחנו נשתמש בחיפוש פשוט כדי להביא הקשר לסוכן
    const relevantChunks = KB.filter(c => 
      c.metadata?.exercise_ids?.includes(exercise_id)
    ).slice(0, 5);
    const kbContext = relevantChunks.map(c => c.text).join("\n");

    // 2. בניית מערך ההודעות ל-AI
    let messages = [
      { 
        role: "system", 
        content: `${SYSTEM_PROMPT}\n\nחוקי ליווי: היה אמפתי מאוד. תן תיקוף (validation) לרגשות המשתמש. השתמש בידע מה-KB המצורף כדי להציע תובנות או תרגילים. אל תענה תשובות קצרות מדי. דבר בעברית.` 
      },
      { 
        role: "system", 
        content: `חומר תיאורטי רלוונטי:\n${kbContext}` 
      }
    ];

    // הוספת היסטוריית השיחה (הזיכרון)
    if (chat_history && Array.isArray(chat_history)) {
      chat_history.forEach(msg => {
        if (msg.role && msg.content) messages.push(msg);
      });
    }

    // הוספת ההודעה הנוכחית של המשתמש
    messages.push({ role: "user", content: user_message });

  // 3. קריאה ל-OpenAI
    const completion = await client.chat.completions.create({
      model: MODEL,
      messages: messages,
      max_completion_tokens: 800
    });

    const assistantText = completion.choices[0].message.content || "";

    // 4. החזרת תשובה לבאבל
    return res.json({
      assistant_message: assistantText
    });

  } catch (err) {
    console.error("Server Error:", err);
    return res.status(500).json({ error: "Server error", detail: err.message });
  }
});

// השורה שסוגרת את הקובץ ומפעילה אותו
app.listen(PORT, () => {
  console.log(`TANGO AI server listening on port ${PORT}`);
});


