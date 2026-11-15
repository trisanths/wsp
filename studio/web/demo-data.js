// Seed data and agent narration for demo mode. mock.js builds the fixed demo objects from this and
// plays the live run. Nothing here talks to a network. Scores in simulated runs are
// simulated for a 1.24B host (Llama-3.2-1B-Instruct). The language run uses the measured numbers from
// results/README.md. The one Demo pill in the top bar labels the whole app.

import { DEMO, storageKey } from './demo-flag.js';
import { addRecent, getRecent, patchRecent } from './util.js';

export const DEMO_IDS = {
  specMath: 'sp_demo_math',
  specLegal: 'sp_demo_legal',
  specShrink: 'sp_demo_code',
  specSolver: 'sp_demo_solver',
  specClause: 'sp_demo_clause',
  specLang: 'sp_demo_lang',
  runLang: 'run_demo_lang',
  runLive: 'run_demo_live',
  runDone: 'run_demo_done',
  runEmpty: 'run_demo_empty',
  builder: 'b_demo',
  shrink: 'sh_demo',
  agent: 'planner-1',
};
export const DEMO_SHRINK_JOB = 'j_demo_code';
export const SOLVER_EVALS = 40;

/* ---------- spec looks ---------- */
const GUARD_IF = { id: 'if_guard', name: 'IFEval', kind: 'guard', floor: 0.45, base: 0.55, recommended: true };
const GUARD_MMLU = { id: 'mmlu_guard', name: 'MMLU', kind: 'guard', floor: 0.4, base: 0.446, recommended: true };

const sc = (model, params_b, score, sourced, url) => ({ model, params_b, score, sourced, source_url: sourced ? url : null });

export const MATH_LOOK = {
  title: 'Phone math tutor',
  summary: 'Solves grade-school word problems and shows each step. Small enough to run on a phone.',
  budget: { evals: 80, compute_min: 90, min_evals: 5, max_evals: 120 },
  size: { max_params_b: 2.0, min_b: 0.5, max_b: 8.0, host_params_b: 1.24 },
  limits: { max_total_params_b: 2.0, max_run_params_b: 1.9, max_heal_min: 20 },
  benchmarks: [
    { id: 'gsm8k', name: 'GSM8K', kind: 'target', target: 0.68, base: 0.58, lora: 0.63, recommended: true, research: true,
      reference_scores: [
        sc('Qwen3-1.7B', 1.7, 0.75, true, 'https://arxiv.org/abs/2505.09388'),
        sc('Llama-3.2-3B', 3, 0.777, true, 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct'),
        sc('Gemma-3-1B', 1, 0.628, true, 'https://arxiv.org/abs/2503.19786'),
        sc('SmolLM2-1.7B', 1.7, 0.48, false),
      ] },
    { ...GUARD_IF, research: true, reference_scores: [sc('Llama-3.2-3B', 3, 0.77, true, 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct'), sc('Gemma-3-1B', 1, 0.8, true, 'https://arxiv.org/abs/2503.19786')] },
    { ...GUARD_MMLU, research: true },
  ],
  extraIngredients: ['qmath_l18_contrast512'],
};

export const LEGAL_LOOK = {
  title: 'Contract reader',
  summary: 'Reads contract clauses and answers questions about them. Keeps its general skills.',
  budget: { evals: 60, compute_min: 75, min_evals: 5, max_evals: 120 },
  size: { max_params_b: 2.0, min_b: 0.5, max_b: 8.0, host_params_b: 1.24 },
  limits: { max_total_params_b: 2.0, max_run_params_b: 1.9, max_heal_min: 20 },
  benchmarks: [
    { id: 'mmlupro:law', name: 'MMLU-Pro law', kind: 'target', target: 0.22, base: 0.12, lora: 0.17, recommended: true },
    { ...GUARD_IF },
    { ...GUARD_MMLU },
  ],
  extraIngredients: [],
};

export const CODE_SHRINK_SPEC = {
  title: 'Code-only Qwen3 model',
  prompt: 'Keep the code skill of Qwen/Qwen3-30B-A3B-Instruct-2507, under 17B',
  summary: 'Keeps the code skill of Qwen3-30B-A3B-Instruct-2507 and removes experts the code skill does not use. The result has to fit under 17B stored.',
  keep_experts: 64,
  heal_steps: 300,
  limits: { max_total_params_b: 17, max_run_params_b: 3.5, max_heal_min: 20 },
  benchmarks: [
    { id: 'humaneval', name: 'HumanEval', kind: 'target', target: 0.85, base: 0.915, recommended: true },
    { id: 'mbpp', name: 'MBPP', kind: 'target', target: 0.75, base: 0.844, recommended: true },
  ],
};

/* ---------- benchmark research ---------- */
export function mathResearch() {
  const steps = [
    { t: 0.8, kind: 'think', text: 'I will look for benchmarks that test step-by-step arithmetic, and check which ones a 1B model can run here.' },
    { t: 2.2, kind: 'search', text: 'Searched: grade-school math word problem benchmark small language models', provider: 'exa' },
    { t: 3.8, kind: 'read', text: 'GSM8K: Training Verifiers to Solve Math Word Problems', url: 'https://arxiv.org/abs/2110.14168', provider: 'exa' },
    { t: 5.6, kind: 'search', text: 'Searched: GSM8K scores for instruct models under 4B parameters', provider: 'exa' },
    { t: 7.4, kind: 'read', text: 'Llama 3.2 model card', url: 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct', provider: 'exa' },
    { t: 9.0, kind: 'read', text: 'Qwen3 technical report', url: 'https://arxiv.org/abs/2505.09388', provider: 'exa' },
    { t: 10.8, kind: 'read', text: 'Gemma 3 technical report', url: 'https://arxiv.org/abs/2503.19786', provider: 'exa' },
    { t: 12.6, kind: 'think', text: 'GSM8K is the closest test I can run. MATH-500 is the usual paper benchmark, but it is far too hard for a 2B limit and Sapien has no runner for it.' },
    { t: 14.4, kind: 'search', text: 'Searched: instruction following benchmark to guard against losing general skills', provider: 'exa' },
    { t: 16.2, kind: 'read', text: 'Instruction-Following Evaluation for Large Language Models', url: 'https://arxiv.org/abs/2311.07911', provider: 'exa' },
    { t: 18.0, kind: 'pick', text: 'Picked GSM8K: it matches the request directly' },
    { t: 19.2, kind: 'pick', text: 'Picked IFEval: catches a build that stops following instructions' },
    { t: 20.4, kind: 'pick', text: 'Picked MMLU: catches a build that loses general knowledge' },
  ];
  const proposals = [
    { name: 'gsm8k', label: 'GSM8K', measures: '8,500 grade-school word problems that need two to eight steps.', why: 'The closest runnable test of a tutor that shows its work. Public scores exist for small models, so the target has a fair reference.', runnable_id: 'gsm8k', kind: 'target', suggested_target: 0.68, in_spec: true,
      reference_scores: MATH_LOOK.benchmarks[0].reference_scores },
    { name: 'if_guard', label: 'IFEval', measures: 'Short instructions with checkable rules, such as length and format.', why: 'A math tune can break instruction following. This guard shows if that happened.', runnable_id: 'if_guard', kind: 'guard', suggested_target: null, in_spec: true,
      reference_scores: MATH_LOOK.benchmarks[1].reference_scores },
    { name: 'mmlu_guard', label: 'MMLU', measures: 'Multiple-choice questions across 57 school and college subjects.', why: 'A cheap check that general knowledge is still there after the new parts go in.', runnable_id: 'mmlu_guard', kind: 'guard', suggested_target: null, in_spec: true,
      reference_scores: [sc('Llama-3.2-3B', 3, 0.63, true, 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct'), sc('Qwen3-1.7B', 1.7, 0.62, false)] },
    { name: 'math_500', label: 'MATH-500', measures: 'Competition problems, much harder than grade school.', why: 'Used in most papers on this skill, but Sapien has no runner for it and a 2B model scores near the floor.', runnable_id: null, kind: 'target', suggested_target: null, in_spec: false,
      reference_scores: [sc('Qwen3-1.7B', 1.7, 0.73, true, 'https://arxiv.org/abs/2505.09388'), sc('Llama-3.2-3B', 3, 0.48, false)] },
  ];
  const sources = steps.filter((x) => x.kind === 'read').map((x) => ({ title: x.text, url: x.url }));
  return { dur: 21.5, steps, proposals, sources,
    reasoning: 'The request is a math tutor that runs on a phone and shows its work, so the main benchmark should test multi-step word problems directly. GSM8K does, and published scores exist for models of 1B to 3B parameters, which gives the target slider a fair reference.\n\nA math tune can cost general skills, so IFEval and MMLU go in as guards with floors just under the host score. MATH-500 is the usual paper benchmark, but it is too hard at this size and has no runner yet, so it stays as reference.' };
}

export function legalResearch() {
  const steps = [
    { t: 0.8, kind: 'think', text: 'I will look for legal reasoning benchmarks that a 1B model can run, and for scores of other small models.' },
    { t: 2.4, kind: 'search', text: 'Searched: legal reasoning benchmark small language models', provider: 'exa' },
    { t: 4.6, kind: 'read', text: 'LegalBench: a collaboratively built benchmark for legal reasoning', url: 'https://arxiv.org/abs/2308.11462', provider: 'exa' },
    { t: 7.0, kind: 'read', text: 'CUAD: an expert-annotated NLP dataset for legal contract review', url: 'https://arxiv.org/abs/2103.06268', provider: 'exa' },
    { t: 9.4, kind: 'search', text: 'Searched: MMLU-Pro law accuracy for models under 4B parameters', provider: 'exa' },
    { t: 11.8, kind: 'read', text: 'MMLU-Pro: a more robust and challenging multi-task benchmark', url: 'https://arxiv.org/abs/2406.01574', provider: 'exa' },
    { t: 14.2, kind: 'think', text: 'LegalBench and CUAD match contract reading best, but Sapien has no runner for either. The law subset of MMLU-Pro is the closest test I can run today.' },
    { t: 16.6, kind: 'search', text: 'Searched: instruction following guard for small instruct models', provider: 'exa' },
    { t: 18.8, kind: 'read', text: 'Instruction-Following Evaluation for Large Language Models', url: 'https://arxiv.org/abs/2311.07911', provider: 'exa' },
    { t: 21.2, kind: 'pick', text: 'Picked MMLU-Pro law: the closest runnable legal test' },
    { t: 22.6, kind: 'pick', text: 'Picked IFEval: guards instruction following' },
    { t: 23.8, kind: 'pick', text: 'Picked MMLU: guards general knowledge' },
  ];
  const proposals = [
    { name: 'mmlupro_law', label: 'MMLU-Pro law', measures: 'Hard multiple-choice law questions with ten options each.', why: 'The closest test Sapien can run for reading legal text. Ten options make guessing less useful.', runnable_id: 'mmlupro:law', kind: 'target', suggested_target: 0.22, in_spec: true,
      reference_scores: [sc('Qwen3-1.7B', 1.7, 0.2, false), sc('Llama-3.2-3B', 3, 0.16, false), sc('Gemma-3-1B', 1, 0.11, false)] },
    { name: 'if_guard', label: 'IFEval', measures: 'Short instructions with checkable rules.', why: 'A legal tune can break instruction following. This guard shows if that happened.', runnable_id: 'if_guard', kind: 'guard', suggested_target: null, in_spec: true,
      reference_scores: [sc('Llama-3.2-3B', 3, 0.77, true, 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct')] },
    { name: 'mmlu_guard', label: 'MMLU', measures: 'Multiple-choice questions across 57 subjects.', why: 'A cheap check that general knowledge survives.', runnable_id: 'mmlu_guard', kind: 'guard', suggested_target: null, in_spec: true, reference_scores: [] },
    { name: 'legalbench', label: 'LegalBench', measures: '162 legal reasoning tasks written by lawyers.', why: 'The standard legal benchmark, but Sapien has no runner for it yet.', runnable_id: null, kind: 'target', suggested_target: null, in_spec: false,
      reference_scores: [sc('Llama-3.2-3B', 3, 0.5, false)] },
    { name: 'cuad', label: 'CUAD', measures: 'Finds 41 kinds of clauses in real contracts.', why: 'Matches contract review well, but it needs span extraction that Sapien cannot score yet.', runnable_id: null, kind: 'target', suggested_target: null, in_spec: false, reference_scores: [] },
  ];
  const sources = steps.filter((x) => x.kind === 'read').map((x) => ({ title: x.text, url: x.url }));
  return { dur: 25, steps, proposals, sources,
    reasoning: 'Reading a contract is closest to legal reasoning questions, so the main benchmark should be a law test. MMLU-Pro law is the nearest one Sapien can run, and its ten answer options make a lucky guess less likely.\n\nLegalBench and CUAD are better matches in spirit, but neither has a runner here, so they stay as reference. IFEval and MMLU guard against a build that learns law and forgets everything else.' };
}


/* ---------- language write-in: measured numbers, results/README.md "Language write-in (2026-11-15)" ---------- */
// Order: Arabic, Japanese, Chinese, Korean, Hindi, English. Belebele test, 900 items per language, never trained on.
export const LANG = {
  codes: ['ar', 'ja', 'zh', 'ko', 'hi', 'en'],
  names: { ar: 'Arabic', ja: 'Japanese', zh: 'Chinese', ko: 'Korean', hi: 'Hindi', en: 'English' },
  base: [42.8, 45.8, 54.4, 45.7, 38.1, 58.7],
  graft: [75.7, 76.0, 84.2, 77.3, 67.3, 89.7],
  scrambled: [55.7, 57.4, 68.3, 57.6, 50.0, 77.4],
  qwen15: [68.4, 68.6, 79.4, 70.8, 48.9, 82.4],
  qwen3: [81.9, 79.7, 86.6, 81.3, 61.7, 90.4],
  donor: [86.0, 84.9, 89.6, 85.7, 69.8, 92.4],
  goal: [0.70, 0.70, 0.75, 0.70, 0.65],
  extra: { French: [52.2, 86.7], Spanish: [51.2, 85.0], Bengali: [34.0, 54.9], Swahili: [36.0, 61.2] },
  graft_b: 5.21,
  mmlu: [44.6, 59.8],
};
export const LANG_SLICE_ID = 'xfam_qwen25_7b_l4_20';
const LANG_TARGET_IDS = LANG.codes.slice(0, 5).map((c) => 'belebele:' + c);
export const langBenchmarks = () => [
  ...LANG.codes.slice(0, 5).map((c, i) => ({ id: 'belebele:' + c, name: 'Belebele ' + LANG.names[c], kind: 'target', target: LANG.goal[i], base: LANG.base[i] / 100, lora: null, recommended: true, research: true, reference_scores: [] })),
  { id: 'belebele:en', name: 'Belebele English', kind: 'guard', floor: 0.55, base: LANG.base[5] / 100, recommended: true, research: true, reference_scores: [] },
  { ...GUARD_MMLU, base: LANG.mmlu[0] / 100, floor: 0.40, research: true, reference_scores: [] },
];
export const LANG_LOOK = {
  title: 'Five new languages for Llama-3.2-1B',
  prompt: 'Teach Llama-3.2-1B to understand Arabic, Japanese, Chinese, Korean and Hindi',
  summary: 'Adds reading comprehension in Arabic, Japanese, Chinese, Korean and Hindi to Llama-3.2-1B-Instruct, using a slice of Qwen2.5-7B-Instruct that already knows them. Scored on Belebele, 900 test items per language, never used for training. English must not get worse.',
  budget: { evals: 24, compute_min: 60, min_evals: 5, max_evals: 120 },
  size: { max_params_b: 6.0, min_b: 0.5, max_b: 8.0, host_params_b: 1.24 },
  limits: { max_total_params_b: 6.0, max_run_params_b: 6.0, max_heal_min: 30 },
  get benchmarks() { return langBenchmarks(); },
  extraIngredients: [],
};
export const LANG_FINDINGS = [
  'The graft adds about 30 points in every language. Arabic goes from 42.8 to 75.7, Japanese from 45.8 to 76.0, Chinese from 54.4 to 84.2, Korean from 45.7 to 77.3 and Hindi from 38.1 to 67.3 (mean of 3 seeds).',
  'Scrambling the donor weights keeps the size and removes most of the gain. The same 5.2B build with scrambled weights scores 55.7, 57.4, 68.3, 57.6 and 50.0, so 17 to 20 points of the gain come from what the donor knows.',
  'French had no training text and still rose from 52.2 to 86.7. Spanish went from 51.2 to 85.0, Bengali from 34.0 to 54.9 and Swahili from 36.0 to 61.2.',
  'It passes native Qwen2.5-1.5B in all six languages. Native Qwen2.5-3B stays 2 to 6 points ahead in four of them, at fewer parameters per token (3.09B against 5.21B).',
  'English improves too. Belebele English goes from 58.7 to 89.7 and MMLU from 44.6 to 59.8, so adding languages did not cost the guard.',
  'The LoRA and full fine-tune arms are still being scored, so this run does not say yet how the graft compares with them.',
];
export function langResearch() {
  const steps = [
    { t: 0.8, kind: 'think', text: 'The request names five languages. I need a reading test that covers all of them with the same questions, so scores compare across languages.' },
    { t: 2.4, kind: 'search', text: 'Searched: multilingual reading comprehension benchmark Arabic Japanese Chinese Korean Hindi', provider: 'exa' },
    { t: 4.2, kind: 'read', text: 'The Belebele Benchmark: a parallel reading comprehension dataset in 122 language variants', url: 'https://arxiv.org/abs/2308.16884', provider: 'exa' },
    { t: 6.0, kind: 'search', text: 'Searched: Llama 3.2 supported languages', provider: 'exa' },
    { t: 7.8, kind: 'read', text: 'Llama 3.2 model card', url: 'https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct', provider: 'exa' },
    { t: 9.6, kind: 'think', text: 'The card lists eight supported languages. Arabic, Japanese, Chinese and Korean are not among them, so the host starts low and a gain will be easy to see.' },
    { t: 11.4, kind: 'search', text: 'Searched: parallel multilingual math benchmark small models', provider: 'exa' },
    { t: 13.2, kind: 'read', text: 'Language Models are Multilingual Chain-of-Thought Reasoners', url: 'https://arxiv.org/abs/2210.03057', provider: 'exa' },
    { t: 15.0, kind: 'read', text: 'Measuring Massive Multitask Language Understanding', url: 'https://arxiv.org/abs/2009.03300', provider: 'exa' },
    { t: 16.8, kind: 'pick', text: 'Picked Belebele in five languages: parallel questions, 900 test items each' },
    { t: 17.8, kind: 'pick', text: 'Picked Belebele English: English must not get worse' },
    { t: 18.6, kind: 'pick', text: 'Picked MMLU: catches a build that loses general knowledge' },
  ];
  const proposals = [
    ...LANG.codes.slice(0, 5).map((c, i) => ({ name: 'belebele_' + c, label: 'Belebele ' + LANG.names[c], measures: 'Multiple-choice reading questions about short passages, 900 test items in ' + LANG.names[c] + '.', why: 'The same questions exist in every language, so a gain in ' + LANG.names[c] + ' is not a change of test.', runnable_id: 'belebele:' + c, kind: 'target', suggested_target: LANG.goal[i], in_spec: true, reference_scores: [] })),
    { name: 'belebele_en', label: 'Belebele English', measures: 'The same passages and questions in English.', why: 'Shows whether English got worse while the new languages came in.', runnable_id: 'belebele:en', kind: 'guard', suggested_target: null, in_spec: true, reference_scores: [] },
    { name: 'mmlu_guard', label: 'MMLU', measures: 'Multiple-choice questions across 57 subjects.', why: 'A cheap check that general knowledge survives.', runnable_id: 'mmlu_guard', kind: 'guard', suggested_target: null, in_spec: true, reference_scores: [] },
    { name: 'mgsm', label: 'MGSM', measures: 'Grade-school math problems translated into ten languages.', why: 'Tests math in other languages, which is a different skill from reading. Sapien has no runner for it yet.', runnable_id: null, kind: 'target', suggested_target: null, in_spec: false, reference_scores: [] },
  ];
  const sources = steps.filter((x) => x.kind === 'read').map((x) => ({ title: x.text, url: x.url }));
  return { dur: 19.5, steps, proposals, sources,
    reasoning: 'Understanding a language is closest to reading comprehension in it. Belebele asks the same questions in every language, so five targets are comparable with each other, and its English version works as a guard.\n\nMGSM tests math in other languages, a different skill, so it stays as reference. MMLU stays as a second guard for general knowledge.' };
}

/* ---------- task profiles for drafted specs ---------- */
// A request is scored against each profile's patterns. The best profile sets the task, the benchmarks and the
// kinds of parts to suggest. Other profiles that match well add their benchmark too.
export const PROFILES = [
  { key: 'lang', task: 'multilingual', pats: [[/spanish|french|german|portuguese|italian|arabic|japanese|chinese|korean|hindi|russian|swahili|bengali|multilingual|bilingual|translat|languages\b/i, 3]], skills: ['multilingual', 'prose_es', 'prose_fr', 'prose_zh', 'prose_ja', 'prose_hi', 'prose_ar'], donor: /qwen2\.5-7b|qwen3|gemma/i, bench: ['mmlu_es'], guards: ['if_guard', 'mmlu_guard'],
    noun: 'language model', summary: (c) => 'Understands and answers in ' + (c.langs.length ? c.langs.join(', ') : 'languages other than English') + ' as well as it does in English. ' + (c.belebele ? 'Scored on Belebele reading comprehension in each language, with English as a guard.' : 'Sapien has no runner for every language, so MMLU in Spanish, which it can score, is the target.'),
    research: { topic: 'multilingual understanding', papers: [['The Belebele Benchmark', 'https://arxiv.org/abs/2308.16884'], ['Measuring Massive Multitask Language Understanding', 'https://arxiv.org/abs/2009.03300']], main: ['MMLU Spanish', 'The MMLU questions in Spanish.', 'The closest runnable test of language understanding. Sapien measured it on this host before.'], ref: ['mgsm', 'MGSM', 'Grade-school math translated into ten languages.', 'Tests math in other languages. Sapien has no runner for it yet.'] } },
  { key: 'code', task: 'code', pats: [[/\b(code|coding|python|rust|sql|typescript|javascript|golang|java|c\+\+|programm\w*|function|compiler|debug\w*|script)\b/i, 3]], skills: ['code'], donor: /coder|code/i, bench: ['humaneval'], guards: ['if_guard', 'mmlu_guard'],
    subs: [[/rust/i, 'Rust', ['code_rust']], [/sql/i, 'SQL', ['sql']], [/typescript|javascript/i, 'TypeScript', ['code_typescript']], [/\bjava\b/i, 'Java', ['code_java']], [/c#|csharp/i, 'C#', ['code_csharp']]],
    noun: 'code helper', summary: (c) => (c.sub ? 'Writes short ' + c.sub + ' code from a plain description. No ' + c.sub + ' score runs here, so HumanEval (Python) stands in as the target.' : 'Writes short Python functions from a plain description and passes unit tests.'),
    research: { topic: 'code generation', papers: [['Evaluating Large Language Models Trained on Code', 'https://arxiv.org/abs/2107.03374'], ['Program Synthesis with Large Language Models', 'https://arxiv.org/abs/2108.07732']], main: ['HumanEval', '164 hand-written Python tasks, each checked by unit tests.', 'The closest runnable test of code writing. It is Python only, so it says little about other languages.'], ref: ['multipl_e', 'MultiPL-E', 'HumanEval and MBPP translated into 18 programming languages.', 'Covers more languages, but Sapien has no runner for it yet.', 'https://arxiv.org/abs/2208.08227'] } },
  { key: 'legal', task: 'legal', pats: [[/legal|law\b|lawyer|contract|clause|attorney|statute/i, 3]], skills: ['law'], donor: /law|saul|legal/i, bench: ['mmlupro:law'], guards: ['if_guard', 'mmlu_guard'],
    noun: 'legal reader', summary: () => 'Reads legal text and answers questions about it. Keeps its general skills.',
    research: { topic: 'legal reasoning', papers: [['LegalBench', 'https://arxiv.org/abs/2308.11462'], ['MMLU-Pro', 'https://arxiv.org/abs/2406.01574']], main: ['MMLU-Pro law', 'Hard multiple-choice law questions with ten options each.', 'The closest legal test Sapien can run.'], ref: ['legalbench', 'LegalBench', '162 legal reasoning tasks written by lawyers.', 'The standard legal benchmark, but Sapien has no runner for it yet.', 'https://arxiv.org/abs/2308.11462'] } },
  { key: 'medicine', task: 'medicine', pats: [[/medic|health|clinic|nurse|triage|patient|doctor|symptom|pharma|diagnos/i, 3]], skills: ['medicine', 'clinical_knowledge', 'clinical_dialogue'], donor: /medic|bio|clinic/i, bench: ['mmlupro:health'], guards: ['if_guard', 'mmlu_guard'],
    noun: 'medical helper', summary: () => 'Answers health and clinical questions in plain language. It is a study aid and not for patient care.',
    research: { topic: 'medical question answering', papers: [['What Disease does this Patient Have? (MedQA)', 'https://arxiv.org/abs/2009.13081'], ['MMLU-Pro', 'https://arxiv.org/abs/2406.01574']], main: ['MMLU-Pro health', 'Hard multiple-choice health and medicine questions with ten options.', 'The closest medical test Sapien can run.'], ref: ['medqa', 'MedQA', 'Medical licensing exam questions.', 'Used in most medical papers, but Sapien has no runner for it yet.', 'https://arxiv.org/abs/2009.13081'] } },
  { key: 'science', task: 'science', pats: [[/chemi|biolog|physic|science|genetic/i, 3]], skills: ['science', 'chemistry', 'biology', 'physics'], donor: /science|bio|chem/i, bench: ['mmlupro:chemistry'], guards: ['if_guard', 'mmlu_guard'],
    subs: [[/chemi/i, 'chemistry', ['chemistry'], 'mmlupro:chemistry'], [/biolog|genetic/i, 'biology', ['biology'], 'mmlupro:biology']],
    noun: 'science tutor', summary: (c) => 'Explains ' + (c.sub || 'science') + ' step by step and answers study questions.',
    research: { topic: 'science question answering', papers: [['Think you have Solved Question Answering? (ARC)', 'https://arxiv.org/abs/1803.05457'], ['MMLU-Pro', 'https://arxiv.org/abs/2406.01574']], main: ['MMLU-Pro', 'Hard multiple-choice questions with ten options, split by subject.', 'The subject split gives a direct test for this request.'], ref: ['gpqa', 'GPQA', 'Graduate-level science questions written by experts.', 'Too hard at this size, and Sapien has no runner for it.', 'https://arxiv.org/abs/2311.12022'] } },
  { key: 'math', task: 'math', pats: [[/math|gsm|arith|algebra|word problem|calculus|geometry/i, 3], [/tutor/i, 0.5]], skills: ['math', 'gsm8k'], donor: /math/i, bench: ['gsm8k'], guards: ['if_guard', 'mmlu_guard'],
    noun: 'math tutor', summary: () => 'Solves grade-school word problems and shows each step.', research: null },
  { key: 'support', task: 'support', pats: [[/support|ticket|routing|customer|helpdesk|help desk|summar|meeting|faq|email|inbox/i, 2.5]], skills: ['instruction', 'decision', 'knowledge'], donor: /instruct/i, bench: ['if_guard_t'], guards: ['mmlu_guard', 'hellaswag'],
    noun: 'support helper', summary: () => 'Follows short instructions to sort, answer or summarize text. No runnable benchmark scores summaries, so IFEval, which checks that the format rules are followed, is the target.',
    research: { topic: 'instruction following', papers: [['Instruction-Following Evaluation for Large Language Models', 'https://arxiv.org/abs/2311.07911'], ['Measuring Massive Multitask Language Understanding', 'https://arxiv.org/abs/2009.03300']], main: ['IFEval', 'Short instructions with rules a program can check, such as length and format.', 'The nearest runnable test of a helper that follows instructions. It does not grade summary quality.'], ref: ['mt_bench', 'MT-Bench', 'Multi-turn questions graded by a stronger model.', 'Needs a judge model, and Sapien has no runner for it yet.', 'https://arxiv.org/abs/2306.05685'] } },
  { key: 'creative', task: 'creative', pats: [[/haiku|poem|poetry|story|stories|lyrics|creative|novel|joke/i, 3]], skills: ['creative_writing'], donor: /./, bench: ['if_guard_t'], guards: ['mmlu_guard', 'hellaswag'],
    noun: 'writer', summary: () => 'Writes short pieces in a set form. No runnable benchmark scores style, so IFEval, which checks form rules such as line counts, is the target. Read a few outputs in the builder to judge the style yourself.',
    research: { topic: 'creative writing with form rules', papers: [['Instruction-Following Evaluation for Large Language Models', 'https://arxiv.org/abs/2311.07911'], ['HellaSwag: Can a Machine Really Finish Your Sentence?', 'https://arxiv.org/abs/1905.07830']], main: ['IFEval', 'Short instructions with rules a program can check, such as length and format.', 'It checks form rules like "three lines", which is the part of poetry a program can score.'], ref: ['creative_writing', 'Creative writing judged by a model', 'Stories and poems graded by a stronger model.', 'Needs a judge model, and Sapien has no runner for it yet.'] } },
  { key: 'knowledge', task: 'knowledge', pats: [[/knowledge|facts|trivia|general|question|q&a|assistant|chat|reasoning/i, 1.5]], skills: ['knowledge', 'reasoning', 'facts'], donor: /./, bench: ['arc', 'mmlu_t'], guards: ['if_guard'],
    noun: 'knowledge model', summary: () => 'Answers general questions and reasons through short problems.',
    research: { topic: 'general knowledge and reasoning', papers: [['Think you have Solved Question Answering? (ARC)', 'https://arxiv.org/abs/1803.05457'], ['Measuring Massive Multitask Language Understanding', 'https://arxiv.org/abs/2009.03300']], main: ['ARC Challenge', 'Grade-school science questions that need a short chain of reasoning.', 'A common test of general reasoning in small models.'], ref: ['gpqa', 'GPQA', 'Graduate-level science questions written by experts.', 'Too hard at this size, and Sapien has no runner for it.', 'https://arxiv.org/abs/2311.12022'] } },
  { key: 'compact', task: 'compact', pats: [[/smaller|faster|compress|lighter|tiny/i, 2]], skills: [], donor: /./, bench: ['arc'], guards: ['if_guard', 'mmlu_guard'], fixedTitle: 'Smaller, faster model', masks: true,
    noun: 'compact model', summary: () => 'Keeps the host at its own size and removes skills the task does not need with deletion masks. Nothing is added, so the model gets faster and no larger. Guards show what the cuts cost.',
    research: { topic: 'compact language models', papers: [['Think you have Solved Question Answering? (ARC)', 'https://arxiv.org/abs/1803.05457'], ['Measuring Massive Multitask Language Understanding', 'https://arxiv.org/abs/2009.03300']], main: ['ARC Challenge', 'Grade-school science questions that need a short chain of reasoning.', 'Shows whether the model still reasons after the cuts.'], ref: ['gpqa', 'GPQA', 'Graduate-level science questions written by experts.', 'Too hard at this size, and Sapien has no runner for it.', 'https://arxiv.org/abs/2311.12022'] } },
];
export const GENERAL_PROFILE = PROFILES.find((p) => p.key === 'knowledge');

/** Scores a request against every profile. Returns { main, extra: [profiles], subs } */
export function classify(prompt) {
  const text = String(prompt || '');
  const scored = PROFILES.map((p, i) => ({ p, i, s: p.pats.reduce((a, [re, w]) => a + (re.test(text) ? w : 0), 0) })).filter((x) => x.s > 0);
  scored.sort((a, b) => b.s - a.s || a.i - b.i);
  if (!scored.length) return { main: GENERAL_PROFILE, extra: [], none: true };
  return { main: scored[0].p, extra: scored.slice(1).filter((x) => x.s >= 2.5 && x.p.key !== 'knowledge' && x.p.key !== 'compact').map((x) => x.p), none: false };
}
/** A short title built from the request itself. */
export function titleFromPrompt(prompt, profile) {
  if (profile && profile.fixedTitle) return profile.fixedTitle;
  let t = String(prompt || '').replace(/\s+/g, ' ').trim().replace(/[.?!]+$/, '');
  t = t.replace(/^(i need|i want|build me|make me|create|give me)\s+/i, '').replace(/^(an?|the|my)\s+/i, '');
  const cut = (re) => { const m = re.exec(t); if (m && m.index > 0) t = t.slice(0, m.index); };
  const generic = /^(small |tiny |compact |little )?(model|assistant|helper|bot|tool|ai)$/i;
  const head0 = t.split(/,| that | which | who /i)[0].replace(/^(small |tiny |compact )/i, '').trim();
  const tail = (/ (?:that|which|who) (.+)$/i.exec(t) || [])[1];
  if (generic.test(head0) && tail) t = tail;
  cut(/,| that | which | who | under | with | and shows | to run /i);
  if (t.split(' ').length > 6) cut(/ for /i);
  t = t.split(' ').slice(0, 6).join(' ').replace(/[,;:]+$/, '').trim();
  if (!t) t = 'New model';
  if (/^(small |tiny )?model$/i.test(t)) t = (profile ? profile.noun : 'model');
  return t.charAt(0).toUpperCase() + t.slice(1);
}

/** Benchmark research script for a profile, in the same shape as mathResearch. */
export function profileResearch(spec, prof) {
  const R = prof.research;
  const main = spec.benchmarks.find((b) => b.kind === 'target') || spec.benchmarks[0];
  const guards = spec.benchmarks.filter((b) => b.kind === 'guard');
  const [mainLabel, mainMeasures, mainWhy] = R.main;
  const steps = [
    { t: 0.8, kind: 'think', text: 'I will look for tests of ' + R.topic + ' that a 1B model can run here, and check which ones have public scores for small models.' },
    { t: 2.4, kind: 'search', text: 'Searched: ' + R.topic + ' benchmark small language models', provider: 'exa' },
    { t: 4.2, kind: 'read', text: R.papers[0][0], url: R.papers[0][1], provider: 'exa' },
    { t: 6.0, kind: 'read', text: R.papers[1][0], url: R.papers[1][1], provider: 'exa' },
    { t: 7.8, kind: 'think', text: mainLabel + ' is the closest test I can run. ' + R.ref[1] + ' is a good match on paper, but Sapien has no runner for it.' },
    { t: 9.6, kind: 'search', text: 'Searched: instruction following benchmark to guard against losing general skills', provider: 'exa' },
    { t: 11.4, kind: 'read', text: 'Instruction-Following Evaluation for Large Language Models', url: 'https://arxiv.org/abs/2311.07911', provider: 'exa' },
    { t: 13.2, kind: 'pick', text: 'Picked ' + main.name + ': it is the closest runnable test of the request' },
    ...guards.slice(0, 2).map((g, i) => ({ t: 14.2 + i * 1, kind: 'pick', text: 'Picked ' + g.name + ': ' + (g.id === 'mmlu_guard' ? 'catches a build that loses general knowledge' : g.id === 'if_guard' ? 'catches a build that stops following instructions' : 'catches a build that loses everyday common sense') })),
  ];
  const sc = (model, params_b, score, url) => ({ model, params_b, score, sourced: true, source_url: url });
  const CARD = 'https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct';
  const REF = { if_guard: [sc('Llama-3.2-3B', 3, 0.774, CARD)], mmlu_guard: [sc('Llama-3.2-3B', 3, 0.634, CARD)], arc: [sc('Llama-3.2-3B', 3, 0.786, CARD)], hellaswag: [sc('Llama-3.2-3B', 3, 0.698, CARD)] };
  const proposals = [
    { name: main.id.replace(/[^a-z0-9]+/g, '_'), label: main.name, measures: mainMeasures, why: mainWhy, runnable_id: main.id, kind: 'target', suggested_target: main.target, in_spec: true, reference_scores: REF[main.id] || [] },
    ...guards.map((g) => ({ name: g.id, label: g.name, measures: g.id === 'mmlu_guard' ? 'Multiple-choice questions across 57 subjects.' : g.id === 'if_guard' ? 'Short instructions with checkable rules, such as length and format.' : 'Everyday sentence completion.', why: 'A tune can cost general skills. This guard shows if that happened.', runnable_id: g.id, kind: 'guard', suggested_target: null, in_spec: true, reference_scores: REF[g.id] || [] })),
    { name: R.ref[0], label: R.ref[1], measures: R.ref[2], why: R.ref[3], runnable_id: null, kind: 'target', suggested_target: null, in_spec: false, reference_scores: [] },
  ];
  const sources = steps.filter((x) => x.kind === 'read').map((x) => ({ title: x.text, url: x.url }));
  return { dur: 15.5, steps, proposals, sources,
    reasoning: 'The request is about ' + R.topic + ', so the main benchmark should test that skill directly. ' + mainLabel + ' does, within what Sapien can run.\n\nA narrow tune can cost general skills, so ' + (guards.map((g) => g.name).join(' and ') || 'a general guard') + ' go in as guards. ' + R.ref[1] + ' is the usual paper benchmark, but Sapien has no runner for it yet, so it stays as reference.' };
}

/* ---------- narration ---------- */
const recentPick = new Map();
// A random item that was not one of the last few picked for this key, so lines do not repeat.
export function choose(key, list) {
  if (!list.length) return null;
  const last = recentPick.get(key) || [];
  const room = Math.min(last.length, Math.max(0, list.length - 1), 3);
  const ban = last.slice(-room);
  let i = Math.floor(Math.random() * list.length);
  for (let k = 0; k < 8 && ban.includes(i); k++) i = Math.floor(Math.random() * list.length);
  last.push(i);
  if (last.length > 4) last.shift();
  recentPick.set(key, last);
  return list[i];
}
const short = (donor) => String(donor || '').split('/').pop();
const layerText = (part) => (part.layers[0] === part.layers[1] ? 'L' + part.layers[0] : 'L' + part.layers[0] + '-' + part.layers[1]);
const p1 = (x) => (x * 100).toFixed(1);
const p0 = (x) => String(Math.round(x * 100));

const TASK_WORD = { math: 'math', code: 'code', legal: 'legal reasoning', medicine: 'medical', multilingual: 'multilingual', science: 'science', general: 'general' };

/** What a planner writes when it starts a new attempt. o: { style, part, at, rank, steps, data, best, host, vs, fitHere, fitAlt, altAt, lastName, lastScore, mask, floor, task } */
export function planThought(o) {
  const { part } = o;
  if (!part) return 'Host-only floor with a ' + o.steps + ' step heal on ' + o.data + ' text. ' + p0(o.host) + '% is the number every part has to beat, so I want it measured on the same items.';
  const D = short(part.donor), P = part.label, DL = layerText(part), task = TASK_WORD[part.task] || 'general';
  const BN = o.bench || 'the target';
  const lead = o.best != null ? 'The leader is at ' + p1(o.best) + '% on ' + BN + ', ' + ((o.best - o.host) * 100).toFixed(1) + ' points over the host. ' : '';
  const fit = o.fitAlt != null ? 'The free fit probe gave ' + o.fitHere.toFixed(2) + ' at L' + o.at + ' against ' + o.fitAlt.toFixed(2) + ' at L' + o.altAt + '. ' : '';
  const by = {
    blocks: [
      lead + P + ' (' + D + ' ' + DL + ') is a whole donor block. ' + fit + 'I place it at L' + o.at + ' with glue rank ' + o.rank + ' and heal ' + o.steps + ' steps on ' + o.data + ' text.',
      o.vs != null ? P + ' from ' + D + ' scored ' + o.vs + ' points over the random control last time, so I keep it and try a heal on ' + o.data + ' text. ' + fit.trim() : P + ' from ' + D + ' has no control score yet. I place it at L' + o.at + ' and heal ' + o.steps + ' steps on ' + o.data + ' text to get one. ' + fit.trim(),
      'Two blocks within 2 layers of each other put the IFEval guard at risk. This time ' + P + ' goes in alone at L' + o.at + ', and the saved parameters pay for a ' + o.steps + ' step heal.',
      lead + 'A whole donor block should carry more of the ' + task + ' skill than a cluster of neurons. ' + D + ' ' + DL + ' at L' + o.at + ' is my next test, glue rank ' + o.rank + '.',
    ],
    small: [
      P + ' adds only ' + part.params_m + 'M parameters, so it fits the size limit with room to spare. ' + D + ' ' + DL + ' reads like ' + task + ' circuitry, so I probe L' + o.at + ' first. The probe is free.',
      (o.lastName ? 'The last small part (' + o.lastName + ') scored ' + p1(o.lastScore) + '%. ' : '') + P + ' targets similar ' + task + ' units at L' + o.at + '. ' + fit.trim(),
      o.vs != null ? lead + P + ' scored ' + o.vs + ' points over the random control, which is the margin I need. I pair it with a ' + o.steps + ' step heal on ' + o.data + ' text so the glue can settle.' : lead + P + ' has not been scored against a control yet. I pair it with a ' + o.steps + ' step heal on ' + o.data + ' text so the glue can settle.',
      'Small parts are cheap to stack. ' + P + ' (' + D + ' ' + DL + ') goes at L' + o.at + ', two layers from the default, to see whether the fit curve is flat there.',
    ],
    masks: [
      o.mask ? 'Removing ' + o.mask + ' knowledge should free capacity for ' + task + '. I pair that deletion mask with ' + P + ' at L' + o.at + ' and check that IFEval stays above its ' + p0(o.floor) + '% floor.' : lead + P + ' at L' + o.at + ' with no mask, to see what the deletion was costing.',
      o.vs != null ? lead + P + ' from ' + D + ' has a control margin of ' + o.vs + ' points. I try it again at L' + o.at + ' ' + (o.mask ? 'with the ' + o.mask + ' mask on top.' : 'with a longer heal.') : lead + P + ' from ' + D + ' goes in at L' + o.at + ' ' + (o.mask ? 'with the ' + o.mask + ' mask on top.' : 'with a ' + o.steps + ' step heal.'),
      'The last masked run kept the guards but gained nothing. ' + P + ' at L' + o.at + ' with glue rank ' + o.rank + ' is a cleaner test of the part alone.',
    ],
  };
  const list = by[o.style] || by.blocks;
  const t = choose('plan.' + o.style, list);
  return o.cross ? 'As a cross-domain check, I try a part from outside this task. ' + t : t;
}

/** A planner's reaction to a finished eval. o: { name, score, leader, collapsed, guard, guardScore, floor, at, healSteps, otherAt } */
export function evalThought(o) {
  if (o.collapsed) {
    return choose('collapse', [
      o.guard + ' fell to ' + p1(o.guardScore) + '%, under its ' + p0(o.floor) + '% floor. The graft at L' + o.at + ' sits too near another part, so I move it three layers away and lower the glue rank.',
      'That one collapsed: ' + o.guard + ' dropped to ' + p1(o.guardScore) + '%. I read it as a placement problem at L' + o.at + ' and not a part problem, so the same part gets another layer.',
      'Guards failed (' + o.guard + ' ' + p1(o.guardScore) + '%, floor ' + p0(o.floor) + '%). The lesson is saved for the analyst. Next I try one part, no mask, heal ' + o.healSteps + ' steps.',
    ]);
  }
  const d = (o.score - o.leader) * 100;
  if (d >= 0) {
    return choose('win', [
      o.name + ' reached ' + p1(o.score) + '%, ' + d.toFixed(1) + ' points over the old leader. The heal loss was still falling, so I will try ' + (o.healSteps * 2) + ' steps next.',
      'New best at ' + p1(o.score) + '%. The guards barely moved, so the part is not hurting general skills. I send the program to the judge for a random-part control.',
    ]);
  }
  return choose('miss', [
    o.name + ' scored ' + p1(o.score) + '%, ' + Math.abs(d).toFixed(1) + ' points under the leader. Not worth a control run, but the placement at L' + o.at + ' goes on the map.',
    p1(o.score) + '% is below the leader by ' + Math.abs(d).toFixed(1) + ' points. The guards held, so I keep the part and change only the layer.',
    o.name + ' trails by ' + Math.abs(d).toFixed(1) + ' points (' + p1(o.score) + '%). A heal of ' + o.healSteps + ' steps may be too short for this part, so I raise it.',
  ]);
}

export function planLine(o) {
  return choose('planline', [
    'Proposed ' + o.count + ' programs for this round',
    o.count + ' new programs queued, ' + (o.refine ? 'one refines the current leader' : 'all from untried placements'),
    'Round ' + o.round + ': ' + o.count + ' programs planned, ' + o.parts + (o.parts === 1 ? ' distinct part' : ' distinct parts'),
  ]);
}

export function probeLine(o) {
  const t = o.fit >= 0.7 ? 'a good fit' : o.fit >= 0.55 ? 'an average fit' : 'a weak fit';
  return choose('probeline', [
    'Checked fit of ' + o.name + ': ' + o.fit.toFixed(2),
    'Fit of ' + o.name + ' is ' + o.fit.toFixed(2) + ', ' + t,
    'Probed ' + o.name + ' at L' + o.at + ': ' + o.fit.toFixed(2),
  ]);
}

export function evalLine(o) {
  const g = o.target;
  if (o.collapsed) return o.name + ' collapsed: ' + o.guard + ' fell to ' + p1(o.guardScore) + '%, under the ' + p0(o.floor) + '% floor. The lesson is saved.';
  if (o.isBest) {
    return choose('evalbest', [
      o.name + ' scored ' + p1(g) + '% on ' + (o.bench || 'the target') + (o.prev != null ? ', ' + ((g - o.prev) * 100).toFixed(1) + ' points above the old leader' : '') + '. New best.',
      'Scored ' + o.name + ' at ' + p1(g) + '% on ' + (o.bench || 'the target') + '. It leads the board.',
    ]);
  }
  return choose('evalmiss', [
    'Scored ' + o.name + ' at ' + p1(g) + '% on ' + (o.bench || 'the target') + '.',
    o.name + ' reached ' + p1(g) + '%, ' + (((o.best || g) - g) * 100).toFixed(1) + ' points under the leader.',
    o.name + ' came in at ' + p1(g) + '% on ' + (o.bench || 'the target') + '. Guards held.',
  ]);
}

export function mapLine(o) {
  const list = [
    'Map v' + o.v + ': ' + o.donor + ' parts near L' + o.at + ' lead, ' + o.good + ' of ' + o.total + ' scored above ' + p0(o.thr) + '%. ' + o.bad + (o.bad === 1 ? ' collapse' : ' collapses') + ' so far' + (o.bad ? ', mostly stacked parts.' : '.'),
    'Placement note: ' + o.partName + ' fits best at L' + o.def + '. Three or more layers away it loses about ' + o.loss + ' points of fit.',
  ];
  if (o.kind === 'heal' && o.heal600 != null && o.heal100 != null) return 'Heal length matters: runs with 600 steps or more average ' + p1(o.heal600) + '% on ' + (o.bench || 'the target') + ' against ' + p1(o.heal100) + '% for runs with 300 steps or fewer.';
  if (o.kind === 'pair' && o.pair != null) return 'Pairs beat singles: programs with two parts average ' + p1(o.pair) + ' and single parts average ' + p1(o.single) + (o.pairNames ? '. The best pair is ' + o.pairNames + '.' : '.');
  return choose('map', list);
}

export function controlLine(o) {
  return choose('control', [
    'Control for ' + o.name + ': random parts of the same size reach ' + p1(o.control) + '%, the real build ' + p1(o.score) + '%. The parts add ' + o.pts + ' points.',
    'The random-part control for ' + o.name + ' says the part adds ' + o.pts + ' points (' + p1(o.score) + '% against ' + p1(o.control) + '%).',
  ]);
}

export function nudgeThought(text) {
  return 'Nudge from you: "' + text + '". I will use it for this attempt.';
}

/* ---------- recents ---------- */
// The sidebar and the Runs page list what was opened in this browser. Demo mode fills them with the demo objects.
export function seedDemoRecents() {
  if (!DEMO) return;
  // reseed when the seed changes; this list holds demo items only (its storage key is demo-only)
  const VERSION = '3';
  let seen = null;
  try { seen = localStorage.getItem('sapien.demo.seed'); } catch (e) { /* storage can be off */ }
  const have = getRecent();
  if (seen === VERSION && have.some((r) => r.id === DEMO_IDS.runLive)) return;
  try { localStorage.removeItem(storageKey('wsp.recent')); localStorage.setItem('sapien.demo.seed', VERSION); } catch (e) { /* storage can be off */ }
  const now = Date.now();
  const min = 60 * 1000;
  // oldest first: addRecent puts each new item at the top. One row per demo object, each with its own title.
  const items = [
    { kind: 'run', id: DEMO_IDS.runEmpty, title: 'Clause checker under 1.3B', status: 'empty', evals: 0, total: 60, ago: 22 * 60 },
    { kind: 'run', id: DEMO_IDS.runDone, title: 'Grade-school math solver', status: 'done', evals: SOLVER_EVALS, total: SOLVER_EVALS, ago: 4 * 60 + 12 },
    { kind: 'shrink', id: DEMO_IDS.shrink, spec: DEMO_IDS.specShrink, title: 'Shrink: ' + CODE_SHRINK_SPEC.title, status: 'done', job: DEMO_SHRINK_JOB, ago: 60 + 8 },
    { kind: 'spec', id: DEMO_IDS.specLegal, title: LEGAL_LOOK.title, ago: 11 },
    { kind: 'run', id: DEMO_IDS.runLang, title: LANG_LOOK.title, status: 'done', evals: LANG_LOOK.budget.evals, total: LANG_LOOK.budget.evals, ago: 38 },
    { kind: 'run', id: DEMO_IDS.runLive, title: MATH_LOOK.title, status: 'running', evals: 32, total: 80, ago: 3 },
  ];
  for (const { ago, ...it } of items) {
    addRecent(it);
    patchRecent(it.kind, it.id, { ts: now - ago * min });
  }
}
