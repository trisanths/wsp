// The techniques the miners and the agent swarm use, in pipeline order. The pipeline page, the mining
// pages and the part windows read this list, so a technique reads the same everywhere.
// stage: where it runs (mine, research, build, test). sponsor: the service that runs it (logos.js SPONSORS).
// anim: the Brain3D mode that shows it (brain3d.js). kinds: the activity kinds or part sources it explains.

export const STAGES = [
  { id: 'request', name: 'Request', line: 'A person describes the model they need. The agent model drafts a spec with limits and targets.', sponsor: 'openai' },
  { id: 'research', name: 'Benchmarks', line: 'A research agent searches the web for benchmarks and published scores, and picks targets a model this size can reach.', sponsor: 'exa' },
  { id: 'mine', name: 'Mining', line: 'Agents read open models layer by layer, find the parts that hold one skill, and test each part against random parts of the same size.', sponsor: 'modal' },
  { id: 'swarm', name: 'Agent swarm', line: 'Planner agents write programs in the Sapien language: a host, parts to graft, skills to remove, a heal. They share what they learn.', sponsor: 'pastdev' },
  { id: 'build', name: 'Compile and heal', line: 'Each program compiles into one network on a GPU. Glue layers learn to translate between the donor and the host.', sponsor: 'modal' },
  { id: 'test', name: 'Score and verify', line: 'The network is scored on the targets, then checked against the host, a LoRA, distillation and random parts of the same size.', sponsor: 'modal' },
  { id: 'result', name: 'Result', line: 'The best program that beats every reference becomes the model, with each part labeled by what it contributes.', sponsor: null },
];

export const TECHNIQUES = [
  // mining: interpretability on the donor's weights
  { id: 'weight-read', stage: 'mine', name: 'Weight reading', sponsor: 'openai', anim: 'mine',
    what: 'An agent reads raw weight statistics in a locked sandbox and guesses what a block of neurons does, for example "layer 22, neurons 3010 to 3400: arithmetic".',
    why: 'Turns a 30B parameter model into a short list of named candidates to test.' },
  { id: 'ablation', stage: 'mine', name: 'Ablation', sponsor: 'modal', anim: 'delete',
    what: 'Switch one candidate part off and score the model on every skill at once.',
    why: 'A part that matters for a skill makes that skill drop when it is off.' },
  { id: 'activation-scan', stage: 'mine', name: 'Activation scan', sponsor: 'modal', anim: 'mine',
    what: 'Run skill text through the donor and rank neurons by how much more they fire on that skill than on general text.',
    why: 'Finds skill-specific neurons without training anything.' },
  { id: 'attribution', stage: 'mine', name: 'Attribution', sponsor: 'modal', anim: 'mine',
    what: 'Rank units by gradient times activation on the skill loss: how much each unit moves the answer.',
    why: 'Catches units that matter but do not fire much.' },
  { id: 'saliency', stage: 'mine', name: 'Expert saliency', sponsor: 'modal', anim: 'mine',
    what: 'For a mixture-of-experts model, weigh each expert by its routing weight times the size of what it writes, summed over skill tokens.',
    why: 'The only ranking that kept code skill when half the experts were removed (run 19).' },
  { id: 'skill-map', stage: 'mine', name: 'Skill maps', sponsor: 'modal', anim: 'mine',
    what: 'Map, layer by layer, where each skill lives in the donor: which layers, heads and experts it uses most.',
    why: 'Tells the swarm where to look before it spends a GPU minute.' },
  { id: 'weight-delta', stage: 'mine', name: 'Weight-difference maps', sponsor: 'modal', anim: 'mine',
    what: 'Compare a fine-tune with its base model and find the weights the fine-tune changed most.',
    why: 'A fine-tune for law or code writes its new skill into a small set of weights.' },
  { id: 'gate', stage: 'mine', name: 'Random-part gate', sponsor: 'modal', anim: 'score',
    what: 'Keep a part only if switching it off hurts its own skill, barely touches the others, and hurts more than random parts of the same size (z-score against 5 random draws).',
    why: 'Most plausible-looking parts fail this, which is the point.' },
  { id: 'shared-memory', stage: 'mine', name: 'Shared memory', sponsor: 'pastdev', anim: 'swarm',
    what: 'Miners and planners write what they find to a shared memory, scoped to one run, and read it before they act.',
    why: 'No agent repeats a dead end another agent already hit.' },
  // research
  { id: 'benchmark-search', stage: 'research', name: 'Benchmark search', sponsor: 'exa', anim: null,
    what: 'Search the web for benchmarks that test the request and for published scores of models near the size limit.',
    why: 'Targets come from what similar models really score, with the page that shows the number.' },
  // swarm and build
  { id: 'program', stage: 'swarm', name: 'Weight-space programs', sponsor: 'openai', anim: 'graft',
    what: 'A planner writes a program: graft part X at host layer N, remove skill Y with a deletion mask, heal on Z text for K steps.',
    why: 'A model is assembled from proven parts instead of trained from scratch.' },
  { id: 'fit-probe', stage: 'build', name: 'Fit probe', sponsor: 'modal', anim: 'graft',
    what: 'Before a full compile, measure how well the donor part\'s inputs and outputs line up with the host at that layer (fit r2). Free, no eval used.',
    why: 'Picks the host layer where a part can be read and written cheaply.' },
  { id: 'graft', stage: 'build', name: 'Grafting', sponsor: 'modal', anim: 'graft',
    what: 'Copy the part\'s weights into the host at the chosen layer, with small glue layers that translate in and out.',
    why: 'The result is one network: no second model runs at inference.' },
  { id: 'deletion-mask', stage: 'build', name: 'Deletion masks', sponsor: 'modal', anim: 'delete',
    what: 'Zero a small set of weights (0.1 to 0.2% of the host) that hold a skill the request does not need.',
    why: 'Removes a skill on purpose, so the model forgets what it should not do.' },
  { id: 'synthesis-heal', stage: 'build', name: 'Synthesis heal', sponsor: 'modal', anim: 'heal',
    what: 'Train only the glue on general text, in stages, and roll back any stage that lowers a guarded skill on the practice set.',
    why: 'Makes the part speak the host\'s language without overwriting what the host knows.' },
  { id: 'expert-removal', stage: 'build', name: 'Expert removal', sponsor: 'modal', anim: 'delete',
    what: 'Keep the top experts per layer for one skill and delete the rest with their router rows, then heal by distillation toward the original.',
    why: 'A 30B model kept code within 4 to 5 points at 16B stored (run 19).' },
  // test
  { id: 'controls', stage: 'test', name: 'Random controls', sponsor: 'modal', anim: 'score',
    what: 'Compile the same program with random parts of the same size and the same heal, and score it the same way.',
    why: 'A result counts only if the real parts beat random ones.' },
  { id: 'verdicts', stage: 'test', name: 'Per-part verdicts', sponsor: 'modal', anim: 'score',
    what: 'Switch each grafted part off in the finished network and label it: contributes, no better than random, or harmful.',
    why: 'Every part in the final model has to earn its place.' },
  { id: 'drift', stage: 'test', name: 'Drift check', sponsor: 'modal', anim: 'score',
    what: 'Measure how far the edited network\'s next-token predictions moved from the host on general text (KL divergence).',
    why: 'Catches edits that quietly damage everything else.' },
  { id: 'comparisons', stage: 'test', name: 'Comparison set', sponsor: 'modal', anim: 'score',
    what: 'Score the host, the host with a LoRA, the host with distillation, random parts, and the best existing model at that size.',
    why: 'A build that does not beat all of them by 2 points is a failed attempt.' },
  { id: 'heldout', stage: 'test', name: 'Held-out check', sponsor: 'modal', anim: 'score',
    what: 'Score the top programs again on items the search never saw.',
    why: 'Separates real gains from programs tuned to the search items.' },
];

export const techniqueById = (id) => TECHNIQUES.find((t) => t.id === id) || null;
export const techniquesFor = (stage) => TECHNIQUES.filter((t) => t.stage === stage);
