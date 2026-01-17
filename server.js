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

// ---- Endpoint ----
app.post("/tango/ai/coach_message", async (req, res) => {
  try {

    // ---- Simple auth: require static token from Bubble ----
    if (MIDDLEWARE_TOKEN) {
      const auth = req.headers["authorization"] || "";
      const ok = auth === `Bearer ${MIDDLEWARE_TOKEN}` || auth === MIDDLEWARE_TOKEN;
      if (!ok) {
        return res.status(401).json({ error: "Unauthorized" });
      }
    }

    const {
      user_id,
      session_id,
      exercise_id,
      step_number,
      chat_intent,
      user_message,
      user_inputs_so_far,
      constraints
    } = req.body || {};

    // Minimal validation
    if (!exercise_id || !chat_intent || !user_message) {
      return res.status(400).json({ error: "Missing required fields: exercise_id, chat_intent, user_message" });
    }
    if (exercise_id !== "emotion_naming_v1") {
      return res.status(400).json({ error: "MVP only supports exercise_id=emotion_naming_v1 for now" });
    }

    // Make sure chunk embeddings exist (MVP)
    await ensureChunkEmbeddings();

    // 1) Forced seeds
    const forcedSeeds = pickForcedSeeds(chat_intent, exercise_id);
    const forcedSeedIds = forcedSeeds.map((c) => c.chunk_id);

    // 2) Vector search (complement)
    const queryText = buildQueryText({ chat_intent, exercise_id, step_number, user_message });
    const qVec = await embedText(queryText);

    const candidates = KB.filter((c) => filterByExercise(c, exercise_id));
    const scored = candidates.map((c) => {
      const v = getChunkVec(c.chunk_id);
      const score = v ? cosineSim(qVec, v) : 0;
      return { chunk: c, score };
    }).sort((a, b) => b.score - a.score);

    const topHits = scored.slice(0, TOP_K_SEARCH);

    const top1Score = topHits[0]?.score ?? 0;
    const strength = classifyRetrievalStrength(top1Score);

    // Only accept additional chunks if >= weak threshold
    const vectorAccepted = topHits
      .filter((h) => h.score >= MIN_SCORE_WEAK)
      .map((h) => ({ chunk_id: h.chunk.chunk_id, score: h.score }));

    // 3) Merge: seeds + up to (TOP_K_FINAL - seeds) unique from vector results
    const final = [];
    const seen = new Set();

    for (const s of forcedSeeds) {
      if (!seen.has(s.chunk_id)) { final.push(s); seen.add(s.chunk_id); }
    }
    for (const h of topHits) {
      if (final.length >= TOP_K_FINAL) break;
      if (h.score < MIN_SCORE_WEAK) break;
      const cid = h.chunk.chunk_id;
      if (seen.has(cid)) continue;
      final.push(h.chunk);
      seen.add(cid);
    }

    // 4) Build prompt for OpenAI Responses API
    const kbContext = buildKbContext(final);

    const ragRules =
      "ידע סגור בלבד. ענה אך ורק על בסיס קטעי הידע (KB) המצורפים. " +
      "אם אין בסיס מספיק ב-KB, אמור שאין לך בסיס מספיק בחומר הקנוני ושאל שאלה מבהירה אחת. " +
      "אל תציג תגיות/IDs/ציטוטי KB למשתמש. " +
      "שמור על טון אמפתי אך מכוון: נרמול קצר -> עצירה/הבחנה -> בחירה/צעד קטן. " +
      "אין ייעוץ זוגי ישיר ואין עבודה זוגית סינכרונית בתוך האפליקציה.";

    const appContext =
      `הקשר אפליקטיבי: exercise_id=${exercise_id}, step=${step_number ?? "?"}, intent=${chat_intent}. ` +
      `מגבלות: ${JSON.stringify(constraints || {})}. ` +
      `קלט קודם: ${JSON.stringify(user_inputs_so_far || {})}.`;

    // Responses API input supports a single string or structured input.
    // We'll use a single input string that includes system/dev/user separation.
    // (Alternatively you can use Chat Completions API messages.)
    const input = [
      `SYSTEM:\n${SYSTEM_PROMPT}`,
      `DEVELOPER:\n${ragRules}\n${appContext}`,
      `KB:\n${kbContext || "(no kb chunks)"}`,
      `USER:\n${user_message}`
    ].join("\n\n");

    const openaiResp = await client.responses.create({
      model: MODEL,
      input
    });

    const assistantText = openaiResp.output_text || "";

    // If retrieval is "none", we still allow answer — but it should follow the RAG rule and ask to clarify.
    return res.json({
      assistant_message: assistantText,
      debug: {
        user_id,
        session_id,
        exercise_id,
        step_number,
        chat_intent,
        forced_seed_chunk_ids: forcedSeedIds,
        retrieval: {
          top1_score: top1Score,
          strength,
          vector_hits: vectorAccepted,
          final_chunks_used: final.map((c) => c.chunk_id)
        },
        openai: {
          model: MODEL,
          response_id: openaiResp.id
        }
      }
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "Server error", detail: String(err?.message || err) });
  }
});

app.listen(PORT, () => {
  console.log(`TANGO AI server listening on http://localhost:${PORT}`);
});
