const OpenAI = require('openai');
const { Pinecone } = require('@pinecone-database/pinecone');
const cheerio = require('cheerio');
const fetch = (...args) => import('node-fetch').then(({ default: fetch }) => fetch(...args));
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
NOTE: CHOOSE correct_option AS NONE, if not sure or any confusion`;
  const ctx = contexts.map(c => `- ${c.text}`).join('\n');
  const opts = options.map(o => `${o.label}. ${o.text}`).join('\n');
  return `${intro}\n\nContext:\n${ctx}\n\nQuestion: ${question}\nOptions:\n${opts}\n\nAnswer:`;
}

function buildValidationPrompt(question, options, answer, contexts) {
  const intro = `You are a validator for CPT/ICD coding questions. Evaluate the proposed answer. Return JSON: {"confidence":<0-1>,"rationale":"<brief reason>","correct_option":"<option a,b,c,d,none>","answer":"value or none"}.
NOTE: CHOOSE correct_option AS NONE, if not sure or any confusion`;
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


function detectCodeType(question) {
    const q = question.toLowerCase();
    if (q.includes('hcpcs')) return 'hcpcs';
    if (q.includes('icd')) return 'icd';
    return 'cpt';
  }

// ---------- Scrape CPT/HCPCS/ICD Descriptor ----------
async function scrapeCode(code, codeType) {
  let url, selector;
  if (codeType === 'hcpcs') {
    url = `https://www.aapc.com/codes/hcpcs-codes/${code}`;
    selector = '#offlongdesc .panel-body .tab-pane b';
  } else if (codeType === 'icd') {
    url = `https://www.aapc.com/codes/icd-10-codes/${code}`;
    selector = '#offlongdesc .panel-body .tab-pane b';
  } else {
    url = `https://www.aapc.com/codes/cpt-codes/${code}`;
    // Summary paragraph follows the <strong>Summary</strong>
    selector = '#cpt_layterms strong:contains("Summary")';
  }

  try {
    const res = await fetch(url, { headers: { 'User-Agent': 'Mozilla/5.0', 'Accept': 'text/html' } });
    if (!res.ok) {
      console.warn(`⚠️ Failed to fetch ${codeType.toUpperCase()} ${code}`);
      return null;
    }

    const html = await res.text();
    const $ = cheerio.load(html);

    if (codeType === 'cpt') {
      const strong = $(selector).first();
      return strong.next('p').text().trim() || null;
    }

    // hcpcs or icd
    return $(selector).first().text().trim() || null;
  } catch (err) {
    console.error(`❌ Error scraping ${codeType.toUpperCase()} ${code}:`, err.message);
    return null;
  }
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

function buildScrapePrompt(question, options, summaries) {
    const intro = `NOTE: DO NOT CHOOSE correct_option AS NONE, SELECT MOST CLOSEST OPTION FROM THE GIVEN OPTIONS (A,B,C,D)

    You are a medical coding assistant. Use the following context to answer the multiple-choice question. Select one option (A, B, C, etc.) with step-by-step reasoning, then give a short explanation AND the final selected option as. 
    NOTE :  YOU SHOULD CHOOSE ONE OPTION BASED ON THE INFO , MOST PROBABLE
    NOTE: DO NOT CHOOSE correct_option AS NONE, SELECT MOST CLOSEST OPTION FROM THE GIVEN OPTIONS (A,B,C,D)`;
  const descText = options.map(o => {
    const desc = summaries[o.label] || 'No description available.';
    return `- ${o.label}. ${o.text}\n  Description: ${desc}`;
  }).join('\n\n');

  const opts = options.map(o => `${o.label}. ${o.text}`).join('\n');
  return `${intro}\n\n${descText}\n\nQuestion: ${question}\nOptions:\n${opts}\n\nAnswer:`;
}


async function answerWithScrape(question, options) {
    // 1. Determine code type
    const codeType = detectCodeType(question);
  
    // 2. Fetch descriptions in parallel
    const summaries = {};
    await Promise.all(options.map(async o => {
      summaries[o.label] = await scrapeCode(o.text.trim(), codeType);
    }));
  
    console.log("summaryyy",summaries)
    // 3. Primary reasoning
    const scrapePrompt = buildScrapePrompt(question, options, summaries);
    const primaryRes = await callModel(PRIMARY_MODEL, scrapePrompt);
  
    // 4. Validation with robust JSON parsing
    const validationContexts = options.map(o => ({ text: `${o.label}. ${o.text}: ${summaries[o.label] || ''}` }));
    const validationPrompt = buildValidationPrompt(
      question,
      options,
      primaryRes.text,
      validationContexts
    );
    const valResRaw = await callModel(VALIDATOR_MODEL, validationPrompt);
  
    let validation;
    try {
      validation = JSON.parse(valResRaw.text);
    } catch {
      let cleanText = valResRaw.text.trim();
      if (cleanText.startsWith('```')) {
        cleanText = cleanText.replace(/^```[a-zA-Z]*\n?/, '');
        cleanText = cleanText.replace(/```$/, '');
      }
      const jsonMatch = cleanText.match(/\{[\s\S]*?\}/);
      if (jsonMatch) {
        try {
          validation = JSON.parse(jsonMatch[0]);
        } catch {
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
    };
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
    let recoveredByScrape = 0;

    // let i=11
  
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
  
      // 1. Primary attempt
      const result = await answerPipeline(q.question, opts);
  
      let validated = result.validation?.correct_option || '';
      const expected = ANS[String(i + 1)].toUpperCase();
  
      const validLabels = opts.map(o => o.label.toUpperCase()); // e.g., ['A', 'B', 'C', 'D']
  
      // Normalize casing
      const validatedNormalized = validated.trim().toUpperCase();
  
      const isValidationBad = !validLabels.includes(validatedNormalized) && validatedNormalized !== 'NONE';
  
      if (isValidationBad) {
        console.log(`⚠️ Validation parsing failed or invalid value ("${validated}"). Assuming NONE.`);
      }
  
      if (validatedNormalized === 'NONE' || isValidationBad) {
        console.log(`⚠️ Model picked NONE (unsure). Trying fallback with scrape...`);
  
        // Fallback attempt
        const scrapeResult = await answerWithScrape(q.question, opts);
  
        const scrapeValidated = (scrapeResult.validation?.correct_option || '').trim().toUpperCase();
  
        if (scrapeValidated === expected) {
          console.log(`✅ Corrected with scrape! Picked ${scrapeValidated}, expected ${expected}`);
          correctCount++;
          recoveredByScrape++;
        } else {
          console.log(`❌ Still incorrect after scrape. Picked ${scrapeValidated}, expected ${expected}`);
          failedQuestions.push(i + 1);
        }
  
      } else {
        if (validatedNormalized === expected) {
          console.log(`✅ Correct! Model picked ${validatedNormalized}, expected ${expected}`);
          correctCount++;
        } else {
          console.log(`❌ Incorrect. Model picked ${validatedNormalized}, expected ${expected}`);
          failedQuestions.push(i + 1);
        }
      }
  
      console.log(`🧠 Answer Text:\n${result.finalAnswer}`);
      console.log(`🛠️ Validator Confidence: ${result.validation?.confidence}`);
     }
  
    console.log(`\n🎯 Accuracy: ${correctCount}/${QUES.length} correct`);
  
    if (recoveredByScrape > 0) {
      console.log(`🔁 Fallback scrape corrected ${recoveredByScrape} questions!`);
    }
  
    if (failedQuestions.length > 0) {
      console.log(`\n❗ Failed Questions: ${failedQuestions.join(', ')}`);
    } else {
      console.log(`\n🏆 All questions answered correctly!`);
    }
  })();
  