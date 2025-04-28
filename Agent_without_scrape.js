const OpenAI = require('openai');
const { Pinecone } = require('@pinecone-database/pinecone');
const { ANS,QUES } = require('./question_answer.js');



const GEMINI_API_KEY = '';  // Replace with your actual API key


// ---------- Configuration ----------
const PINECONE_API_KEY = "";

const PINECONE_CONFIGS = {
  hcpcs: {
    index: 'hcpcs',
    host: 'https://hcpcs-vhze9sb.svc.aped-4627-b74a.pinecone.io',
  },
  icd: {
    index: 'icd',
    host: 'https://icd-vhze9sb.svc.aped-4627-b74a.pinecone.io',
  },
  cpt: {
    index: 'cpt',
    host: 'https://cpt-vhze9sb.svc.aped-4627-b74a.pinecone.io',
  },
};

const NEBIUS_API_KEY = "";

// ---------- LLM Clients ----------
const llmClient = new OpenAI({
  baseURL: 'https://api.studio.nebius.com/v1/',
  apiKey: NEBIUS_API_KEY,
});

// ---------- Embeddings ----------
async function embedText(text) {
  const resp = await llmClient.embeddings.create({
    model: 'BAAI/bge-en-icl',
    input: text,
  });
  return resp.data[0].embedding;
}

// ---------- Dynamic Index Selection ----------
function selectPineconeConfig(question) {
  const qLower = question.toLowerCase();
  if (qLower.includes('hcpcs')) return PINECONE_CONFIGS.hcpcs;
  if (qLower.includes('icd')) return PINECONE_CONFIGS.icd;
  if (qLower.includes('cpt')) return PINECONE_CONFIGS.cpt;
  return PINECONE_CONFIGS.cpt;
}

// ---------- Similarity Query ----------
async function querySimilar(question, k = 35) {
  const pineconeConfig = selectPineconeConfig(question);

  const pc = new Pinecone({ apiKey: PINECONE_API_KEY });
  const index = pc.index(pineconeConfig.index, pineconeConfig.host);

  const embedding = await embedText(question);

  const response = await index.namespace('').query({
    topK: k,
    vector: embedding,
    includeMetadata: true,
  });

  return response.matches.map(m => ({
    id: m.id,
    text: m.metadata.text,
    score: m.score,
  }));
}

// ---------- Prompt Builders ----------
function buildPrompt(question, options, contexts) {
  const intro = `You are a medical coding assistant. Use the following context to answer the multiple-choice question. Select one option (A, B, C, etc.) with step-by-step reasoning, then give a short explanation AND the final selected option.
NOTE: YOU SHOULD CHOOSE ONE OPTION BASED ON THE INFO, MOST PROBABLE.
NOTE: DO NOT CHOOSE correct_option AS NONE, SELECT THE MOST CLOSEST OPTION FROM (A, B, C, D).`;
  const ctx = contexts.map(c => `- ${c.text}`).join('\n');
  const opts = options.map(o => `${o.label}. ${o.text}`).join('\n');
  return `${intro}\n\nContext:\n${ctx}\n\nQuestion: ${question}\nOptions:\n${opts}\n\nAnswer:`;
}

function buildValidationPrompt(question, options, answer, contexts) {
  const intro = `You are a validator for CPT/ICD coding questions. Evaluate the proposed answer. Return JSON: {"confidence":<0-1>,"rationale":"<brief reason>","correct_option":"<option a,b,c,d>","answer":"value"}.
NOTE: DO NOT CHOOSE correct_option AS NONE, SELECT THE MOST CLOSEST OPTION FROM (A, B, C, D).`;
  const ctx = contexts.map(c => `- ${c.text}`).join('\n');
  const opts = options.map(o => `${o.label}. ${o.text}`).join('\n');
  return `${intro}\n\nContext:\n${ctx}\n\nQuestion: ${question}\nOptions:\n${opts}\n\nProposed Answer: ${answer}`;
}

// ---------- New Gemini Util ----------
async function callGemini(prompt) {
  const start = Date.now();

  const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-04-17:generateContent?key=${GEMINI_API_KEY}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{
        parts: [{ text: prompt }]
      }]
    }),
  });

  const data = await response.json();

  const text = data.candidates?.[0]?.content?.parts?.[0]?.text || 'No response';

  return {
    text,
    latency: Date.now() - start,
    usage: {}, // Gemini API doesn't return usage tokens currently
  };
}

// ---------- Model Call (handles Nebius + Gemini) ----------
async function callModel(modelName, prompt) {
  if (modelName === 'gemini') {
    return callGemini(prompt);
  }

  const start = Date.now();
  const res = await llmClient.chat.completions.create({
    model: modelName,
    temperature: 0,
    max_tokens: 2000,
    messages: [{ role: 'user', content: prompt }],
  });

  return {
    text: res.choices[0].message.content,
    latency: Date.now() - start,
    usage: res.usage,
  };
}

// ---------- MCQ Pipeline ----------
const PRIMARY_MODEL = 'gemini';
const VALIDATOR_MODEL = 'microsoft/phi-4';

async function clarifyQuestion(originalQuestion) {
  const prompt = `
You are a helpful medical assistant. Simplify the following medical question so that it is clearer and easier to understand.
Use plain English, avoid technical terms when possible, and explain any complicated procedures briefly. Keep the meaning the same.

Original Question:
${originalQuestion}

Simplified Question:
  `;
  const res = await callModel(PRIMARY_MODEL, prompt);
  return res.text.trim();
}

async function answerPipeline(question, options) {
  const clarifiedQuestion = await clarifyQuestion(question);

  console.log("🔎 Clarified Question:", clarifiedQuestion);

  const contexts = await querySimilar(question);

  const primaryPrompt = buildPrompt(question, options, contexts);
  const primaryRes = await callModel(PRIMARY_MODEL, primaryPrompt);

  const validationPrompt = buildValidationPrompt(
    question,
    options,
    primaryRes.text,
    contexts
  );
  const valResRaw = await callModel(VALIDATOR_MODEL, validationPrompt);

  let validation;

  try {
    validation = JSON.parse(valResRaw.text);
  } catch {
    let cleanText = valResRaw.text.trim();

    if (cleanText.startsWith('```')) {
      cleanText = cleanText.replace(/^```[a-zA-Z]*\n?/, '').replace(/```$/, '');
    }

    const jsonMatch = cleanText.match(/\{[\s\S]*?\}/);

    if (jsonMatch) {
      try {
        validation = JSON.parse(jsonMatch[0]);
      } catch (innerErr) {
        console.error('❌ Still failed to parse inner JSON:', innerErr);
        validation = { confidence: 0, rationale: cleanText };
      }
    } else {
      validation = { confidence: 0, rationale: cleanText };
    }
  }

  return {
    finalAnswer: primaryRes.text,
    primary: primaryRes,
    validation,
    contexts,
  };
}

// ---------- Main Runner ----------
// Example
(async () => {
  let correctCount = 0;
  let failedQuestions = [];

  for (let i = 0; i < QUES.length; i++) {
    const q = QUES[i];

    const opts = q.options.map(optionStr => {
      const [label, ...textParts] = optionStr.split('.');
      return {
        label: label.trim(),
        text: textParts.join('.').trim()
      };
    });

    console.log(`\n📘 Q${i + 1}: ${q.question} ------`);

    const result = await answerPipeline(q.question, opts);

    console.log(result.validation);

    const validated = result.validation?.correct_option?.toUpperCase();
    const expected = ANS[String(i + 1)].toUpperCase();

    if (validated === expected) {
      console.log(`✅ Correct! Model picked ${validated}, expected ${expected}`);
      correctCount++;
    } else {
      console.log(`❌ Incorrect. Model picked ${validated}, expected ${expected}`);
      failedQuestions.push(i + 1);
    }

    console.log(`🧠 Answer Text:\n${result.finalAnswer}`);
    console.log(`🛠️ Validator Confidence: ${result.validation?.confidence}`);
  }

  console.log(`\n🎯 Accuracy: ${correctCount}/${QUES.length} correct`);

  if (failedQuestions.length > 0) {
    console.log(`\n❗ Failed Questions: ${failedQuestions.join(', ')}`);
  } else {
    console.log(`\n🏆 All questions answered correctly!`);
  }
})();
