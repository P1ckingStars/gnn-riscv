# Related work — reading list

*Workstream A deliverable. Filled in as papers are read; status = `[ ]` to read, `[r]` read, `[s]` summarized.*

## Classical superoptimization
- [ ] Massalin, "Superoptimizer: A look at the smallest program" (ASPLOS 1987).
- [ ] Bansal & Aiken, "Automatic generation of peephole superoptimizers" (ASPLOS 2006).
- [ ] Schkufza et al., "Stochastic Superoptimization" / STOKE (ASPLOS 2013).
- [ ] Sasnauskas et al., "Souper: A synthesizing superoptimizer" (2017).
- [ ] Joshi et al., "Denali: A goal-directed superoptimizer" (PLDI 2002).

## Neural superoptimization / program synthesis
- [ ] Bunel et al., "Learning to superoptimize programs" (ICLR 2017).
- [ ] Balog et al., "DeepCoder: Learning to write programs" (ICLR 2017).
- [ ] Devlin et al., "RobustFill" (ICML 2017).
- [ ] Bunel et al., "Leveraging grammar and RL for neural program synthesis" (ICLR 2018).
- [ ] Mankowitz et al., "AlphaDev: Faster sorting algorithms discovered using deep RL" (Nature 2023). **Most directly analogous prior result.**

## GNNs over code / IR
- [ ] Cummins et al., "ProGraML: A graph-based program representation" (ICML 2021).
- [ ] Allamanis et al., "Learning to represent programs with graphs" (ICLR 2018).
- [ ] Brauckmann et al., "Compiler-based graph representations for deep learning models of code" (CC 2020).
- [ ] VenkataKeerthy et al., "IR2Vec" (TACO 2020).

## Generative GNNs (candidate decoder architectures)
- [ ] You et al., "GraphRNN" (ICML 2018).
- [ ] Shi et al., "GraphAF: a Flow-based Autoregressive Model for Molecular Graph Generation" (ICLR 2020).
- [ ] Vignac et al., "DiGress: Discrete Denoising Diffusion for Graph Generation" (ICLR 2023).
- [ ] Goyal et al., "GraphGen" (WWW 2020).

## Learned compiler decisions (for contrast — different problem)
- [ ] Haj-Ali et al., "NeuroVectorizer" (CGO 2020).
- [ ] Trofin et al., "MLGO: A Machine Learning Guided Compiler Optimizations Framework" (2021).
- [ ] Huang et al., "AutoPhase" (MLSys 2020).
- [ ] Zheng et al., "Ansor" (OSDI 2020).
- [ ] Cummins et al., "CompilerGym" (CGO 2022).

## RISC-V backend background
- [ ] Read LLVM RISC-V backend source for the scalar instruction patterns.
- [ ] RV64GC calling convention summary; ABI register usage.
- [ ] Survey what `gcc -O2` actually does on small kernels (peepholes, strength reduction, CSE).

## Synthesis target — one-page "novelty claim"
*To write after the survey is read.* Working hypothesis: first DAG-generative GNN trained
for spec → ISA synthesis with verifier-in-loop on RV64GC, contrasted against seq2seq +
verifier and against pure search.
