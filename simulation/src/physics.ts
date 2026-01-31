/**
 * Ring Fountain — reduced-order physics model (not CFD).
 * Converts user inputs (cm, g) to SI, computes impact speed, focusing coefficient,
 * jet launch speed, and predicted max fountain height.
 */

export const G = 9.81 // m/s²
export const RHO_WATER = 1000 // kg/m³
export const GAMMA = 0.072 // N/m (surface tension)

export type WettingMode = 'hydrophilic' | 'neutral' | 'hydrophobic'

const WETTING_FACTORS: Record<WettingMode, number> = {
  hydrophilic: 1.05,
  neutral: 1.0,
  hydrophobic: 0.95,
}

// Focusing coefficient: C = C0 * eta^a * (1-eta)^b * (sigma*/sigma0)^c * wettingFactor
const C0 = 0.65
const SIGMA0 = 25 // dimensionless reference
const EXP_A = 0.7
const EXP_B = 0.7
const EXP_C = 0.08

export interface InputParams {
  /** Drop height (cm) */
  H_drop_cm: number
  /** Outer diameter (cm) */
  D_o_cm: number
  /** Inner diameter (cm) */
  D_i_cm: number
  /** Ring mass (g) */
  m_g: number
  wetting: WettingMode
}

export interface PhysicsResult {
  /** Impact speed (m/s) */
  v_impact: number
  /** Froude number */
  Fr: number
  /** Weber number */
  We: number
  /** Geometry factor eta = D_i / D_o */
  eta_geom: number
  /** Focusing coefficient C */
  C: number
  /** Jet launch speed (m/s) */
  u_j: number
  /** Predicted max fountain height (cm) */
  H_max_cm: number
}

/** Clamp eta to [0.05, 0.95] */
function clampEta(eta: number): number {
  return Math.max(0.05, Math.min(0.95, eta))
}

/**
 * Compute all physics outputs from input params.
 * Inputs: H_drop (cm), D_o (cm), D_i (cm), m (g). Internal conversions to m and kg.
 */
export function computePhysics(params: InputParams): PhysicsResult {
  const H_drop_m = params.H_drop_cm / 100
  const D_o_m = params.D_o_cm / 100
  // Clamp D_i so that D_i < D_o (e.g. at least 0.5 cm gap)
  const D_i_cm_clamped = Math.min(
    params.D_i_cm,
    Math.max(0.5, params.D_o_cm - 0.5)
  )
  const D_i_m = D_i_cm_clamped / 100
  const m_kg = params.m_g / 1000

  // Impact speed: v = sqrt(2 g H_drop)
  const v_impact = Math.sqrt(2 * G * H_drop_m)

  // Geometry: eta = D_i / D_o, clamped
  const eta_raw = D_o_m > 0 ? D_i_m / D_o_m : 0.5
  const eta_geom = clampEta(eta_raw)

  // Annulus area (m²): A = (π/4) * (D_o² - D_i²)
  const A = (Math.PI / 4) * (D_o_m * D_o_m - D_i_m * D_i_m)
  const sigma = A > 0 ? m_kg / A : 0
  const sigmaStar = D_o_m > 0 ? sigma / (RHO_WATER * D_o_m) : 0

  const wettingFactor = WETTING_FACTORS[params.wetting]
  const C =
    C0 *
    Math.pow(eta_geom, EXP_A) *
    Math.pow(1 - eta_geom, EXP_B) *
    Math.pow(Math.max(sigmaStar / SIGMA0, 0.01), EXP_C) *
    wettingFactor

  const u_j = C * v_impact
  const H_max_m = (u_j * u_j) / (2 * G)
  const H_max_cm = H_max_m * 100

  const Fr = D_o_m > 0 ? (v_impact * v_impact) / (G * D_o_m) : 0
  const We =
    D_o_m > 0
      ? (RHO_WATER * v_impact * v_impact * D_o_m) / GAMMA
      : 0

  return {
    v_impact,
    Fr,
    We,
    eta_geom,
    C,
    u_j,
    H_max_cm,
  }
}

/** Ballistic height at time t (jet tip): h(t) = u_j*t - 0.5*g*t^2. Returns 0 once back at surface. */
export function jetHeightAtTime(u_j: number, t: number): number {
  const h = u_j * t - 0.5 * G * t * t
  return Math.max(0, h)
}

/** Time when jet tip returns to surface: u_j*t - 0.5*g*t^2 = 0 => t = 2*u_j/g */
export function jetTimeOfFlight(u_j: number): number {
  return (2 * u_j) / G
}
