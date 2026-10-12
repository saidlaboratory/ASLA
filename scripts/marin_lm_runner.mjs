// Runs the Marin speedrun leaderboard's own fit (website/src/utils/regression.ts, unmodified) on given points.
//
// Usage: node scripts/marin_lm_runner.mjs <path to regression.ts> < input.json > output.json
// Input:  {"fits": [{"id": "...", "points": [{"x": .., "y": ..}], "targets": [x, ...]}]}
// Output: {"fits": [{"id": "...", "params": {intercept, slope, asymptote, r2} | null, "predictions": [..]}]}
// regression.ts imports ml-levenberg-marquardt, which must resolve from its directory.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const regression = await import(pathToFileURL(process.argv[2]).href);
const input = JSON.parse(readFileSync(0, "utf8"));
const out = input.fits.map(({ id, points, targets }) => {
  const params = regression.computePowerLawWithOffset(points);
  return {
    id,
    params,
    predictions: params ? targets.map((x) => regression.predictWithPowerLaw(params, x)) : targets.map(() => null),
  };
});
process.stdout.write(JSON.stringify({ fits: out }));
