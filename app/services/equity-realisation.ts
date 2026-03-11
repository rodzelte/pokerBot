/**
 * equity-realisation.ts
 *
 * Adjusts raw equity based on how well a hand can realise
 * its equity across future streets.
 */

import { BoardTexture } from "./hand-range-utils.ts";

export type HandClass =
  | "monster"
  | "strongMade"
  | "weakMade"
  | "strongDraw"
  | "weakDraw"
  | "air";

export function calculateEquityRealisation(
  rawEquity: number,
  handClass: HandClass,
  inPosition: boolean,
  texture: BoardTexture,
  isAggressor: boolean
): number {

  let eqr = 1.0;

  // Position advantage
  if (inPosition) eqr += 0.10;
  else eqr -= 0.10;

  // Initiative advantage
  if (isAggressor) eqr += 0.05;

  // Hand category adjustments
  switch (handClass) {

    case "monster":
      eqr += 0.10;
      break;

    case "strongMade":
      eqr += 0.05;
      break;

    case "weakMade":
      eqr -= 0.05;
      break;

    case "strongDraw":
      eqr += 0.12;
      break;

    case "weakDraw":
      eqr -= 0.05;
      break;

    case "air":
      eqr -= 0.10;
      break;
  }

  // Wet boards favour draws
  if (texture.wetScore >= 3) {
    if (handClass === "strongDraw") eqr += 0.05;
  }

  const realisedEquity = rawEquity * eqr;

  return Math.max(0, Math.min(1, realisedEquity));
}
