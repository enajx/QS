# Quorum Sensing (QS)

Simulate bacteria that collectively recognize their spatial shape via quorum sensing. Optimize GRN parameters so colonies express GFP (green) for circle shapes, RFP (red) for star shapes.

## Model

Each cell runs a GRN (ODEs with Hill functions). Cells produce signaling molecules (AHL) that diffuse extracellularly, coupling the colony.

```
dX/dt = α * Hill(A, K, n) - δ * X      (intracellular)
∂C/∂t = D * ∇²C - μ * C + Σ sources    (extracellular AHL)
```

**Trainable**: Hill coefficients `n`, thresholds `K`, production rates `α`
To train it we do: 
1. Simulate quorum sensing programation in colony of cells by solving GRN equations
2. Compare final expression levels to target pattern (circle/star)
3. Optimise the free parameters of the models to produce the desired pattern (either via gradient or CMA-ES)

## Structure

```
src/models/
├── QS_GRN.py    # GRN dynamics per cell
├── colony.py    # Colony simulation (cells + diffusion)
├── shapes.py    # Circle/star generators
└── train.py     # PyTorch optimization
configs/
results/
```

## Techstack

SciPy (ODEs/PDEs), PyTorch (optimization), NumPy


# Code Style Guidelines

- **CRITICAL: NEVER rewrite or create "simplified" versions of existing code when encountering compilation errors. ALWAYS fix the compilation errors in the existing implementation instead. Do not ever create alternative implementations.**
- NEVER use default values in function parameters
- ONLY write what requested explicitly, do not add extra stuff
- Always require all parameters to be explicitly passed
- This ensures clarity and prevents ordering issues
- Be concise, do not over-comment the code
- Avoid try/except logic unless very explicitly asked
- Avoid using .get() methods with default values
- DO NOT run bash commands to evaluate code changes unless told to "verify"
- Commits messags should very concise and not include claude code ads.