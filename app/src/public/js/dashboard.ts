import { diffWords } from "diff";

function parseMessage(message: string) {
  const iterativeImprovementsRegex = /I-\d+ +(.+)/g;
  let match = iterativeImprovementsRegex.exec(message);
  const numIterativeImprovements = match[1];

  const stepsRegex = /E-\d+_\d+ +(.+)/g;
  match = stepsRegex.exec(message);
  const steps = [match[1]];
  while (match != null) {
    match = stepsRegex.exec(message);
    if (match != null) {
      steps.push(match[1]);
    }
  }
  return { numIterativeImprovements, steps };
}

function visualizeDiff() {
  const message = `
    S-272120        german stocks closed down wednesday at the frankfurt stock exchange .
    T-272120        german stocks close lower
    H-272120        -0.9572659730911255     german stocks close down at frankfurt bourse
    P-272120        0.0000 -0.3565 -0.7092 -0.7942 -0.9684 -0.2987 -1.5121 -3.9764 0.0000
    I-272120        2
    E-272120_0
    E-272120_1      <unk> <unk> <unk> <unk> <unk> <unk>
    E-272120_2      german stocks close down at frankfurt
    E-272120_3      german stocks close down at frankfurt
    E-272120_4      german stocks close down at frankfurt <unk>
    E-272120_5      german stocks close down at frankfurt bourse
    E-272120_6      german stocks close down at frankfurt bourse
    E-272120_7      german stocks close down at frankfurt bourse
    E-272120_8      german stocks close down at frankfurt bourse
    `;
  const { numIterativeImprovements, steps } = parseMessage(message);
  if (steps.length < 2) {
    console.error(
      "Not enough iterative improvements for visualization, aborting."
    );
    return;
  }
  const display = document.getElementById("levenshtein-vizualization");
  for (let stepIdx = 1; ++stepIdx; stepIdx < steps.length) {
    const nextSentence = steps[stepIdx];
    const oldSentence = steps[stepIdx - 1];
    const diff = diffWords(oldSentence, nextSentence);
    const fragment = document.createDocumentFragment();
    diff.forEach((part: any) => {
      const color = part.added ? "green" : part.removed ? "red" : "grey";
      const span = document.createElement("span");
      span.style.color = color;
      span.appendChild(document.createTextNode(part.value));
      fragment.appendChild(span);
    });
    display.appendChild(fragment);
  }
}

$(document).ready(() => {
  console.log(12);
  document.getElementById("start-demo").onclick = visualizeDiff;
});
