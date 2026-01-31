# Ring Fountain Simulator

A single-page web app that simulates a **ring fountain**: a flat metal ring dropped onto a water surface produces a vertical Worthington-like jet. This is a **reduced-order model** (not CFD)—it uses algebraic and scaling relations to predict jet height and dimensionless numbers from drop height, ring geometry, mass, and a wetting factor.

## Run locally

```bash
cd simulation
npm install
npm run dev
```

Then open **http://localhost:5173** in your browser.

## What it does

- **Inputs (right panel):** Drop height \(H_{drop}\), outer/inner diameter \(D_o\), \(D_i\), ring mass \(m\), and a wetting mode (hydrophilic / neutral / hydrophobic).
- **Run Simulation:** Computes impact speed, Froude number, Weber number, geometry factor, and predicted max fountain height \(H_{max}\).
- **3D view:** A tank of water, a ring falling from the set height, and a tapered jet that rises and falls following ballistic motion, with slight wobble and ripples on the water surface.
- **Recent runs table (top-left):** Last 10 runs with \(H_{drop}\), \(D_o\), \(D_i\), \(m\), and \(H_{max}\).

## Model (reduced-order, not CFD)

- **Impact speed:** \(v = \sqrt{2 g H_{drop}}\) (SI: m, m/s).
- **Geometry:** \(\eta = D_i / D_o\) (clamped), annulus area \(A = (\pi/4)(D_o^2 - D_i^2)\), areal density \(\sigma = m/A\), dimensionless \(\sigma^* = \sigma / (\rho_{water} D_o)\).
- **Focusing coefficient:** \(C = C_0 \, \eta^a (1-\eta)^b (\sigma^*/\sigma_0)^c \times \text{wettingFactor}\) (with defaults \(C_0=0.65\), \(a=b=0.7\), \(c=0.08\), \(\sigma_0=25\); wetting: hydrophilic 1.05, neutral 1.0, hydrophobic 0.95).
- **Jet speed and height:** \(u_j = C v\), \(H_{max} = u_j^2/(2g)\) (output in cm).
- **Froude:** \(\mathrm{Fr} = v^2/(g D_o)\); **Weber:** \(\mathrm{We} = \rho v^2 D_o / \gamma\) (\(\gamma = 0.072\) N/m).

All unit conversions (cm→m, g→kg) are applied inside the physics module.

## Stack

- **React 18** + **TypeScript** + **Vite**
- **Three.js** via **@react-three/fiber** and **@react-three/drei** for the 3D tank, water, ring, jet, and ripples
- **Tailwind CSS** for layout and styling

No backend; everything runs in the browser.
