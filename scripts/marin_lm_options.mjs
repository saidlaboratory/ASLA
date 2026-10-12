// The leaderboard's computePowerLawWithOffset (marin-community/speedrun website/src/utils/regression.ts),
// re-implemented with a configurable iteration cap and initial values. At the defaults it must match the
// unmodified function exactly; the "reference" field carries that function's output for the check.
//
// Usage: node scripts/marin_lm_options.mjs <dir with regression.ts and node_modules> < input.json
// Input:  {"fits": [{"id", "points": [{x, y}], "targets": [x], "maxIterations": n, "cInit": c|null, "bInit": b|null}]}
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import path from "node:path";

const dir = path.resolve(process.argv[2]);
const require = createRequire(path.join(dir, "package.json"));
const { levenbergMarquardt } = require("ml-levenberg-marquardt");
const reference = await import(pathToFileURL(path.join(dir, "regression.ts")).href);

function fit(points, maxIterations, cOverride, bOverride) {
  const valid = points.filter((p) => p.x > 0 && Number.isFinite(p.x) && Number.isFinite(p.y)).sort((a, b) => a.x - b.x);
  if (valid.length < 3) return null;
  const xData = valid.map((p) => p.x);
  const yData = valid.map((p) => p.y);
  const yMin = Math.min(...yData);
  const cInit = cOverride ?? Math.min(0.7, yMin * 0.7);
  const adjusted = valid.map((p) => ({ x: p.x, y: p.y - cInit })).filter((p) => p.y > 0);
  let aInit = 1;
  let bInit = 0.1;
  if (adjusted.length >= 2) {
    const reg = reference.computeLinearRegression(adjusted.map((p) => ({ x: Math.log(p.x), y: Math.log(p.y) })));
    if (reg) {
      bInit = Math.max(0.01, -reg.slope);
      aInit = Math.exp(reg.intercept);
    }
  }
  if (bOverride !== null && bOverride !== undefined) bInit = bOverride;
  const f = ([a, b, c]) => (x) => a * Math.pow(x, -b) + c;
  try {
    const result = levenbergMarquardt({ x: xData, y: yData }, f, {
      initialValues: [aInit, bInit, cInit],
      damping: 1.5,
      gradientDifference: 1e-6,
      maxIterations,
      errorTolerance: 1e-8,
      minValues: [1e-10, 1e-6, 0],
      maxValues: [1e30, 2, yMin * 0.999],
    });
    const [a, b, c] = result.parameterValues;
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c) || a <= 0 || b <= 0 || c < 0) return null;
    return { intercept: a, slope: -b, asymptote: c, iterations: result.iterations };
  } catch {
    return null;
  }
}

const input = JSON.parse(readFileSync(0, "utf8"));
const out = input.fits.map(({ id, points, targets, maxIterations, cInit, bInit }) => {
  const params = fit(points, maxIterations ?? 200, cInit ?? null, bInit ?? null);
  const ref = reference.computePowerLawWithOffset(points);
  return {
    id,
    params,
    predictions: targets.map((x) => (params ? reference.predictWithPowerLaw(params, x) : null)),
    reference_predictions: targets.map((x) => (ref ? reference.predictWithPowerLaw(ref, x) : null)),
  };
});
process.stdout.write(JSON.stringify({ fits: out }));
