# GreenServe IPW Plan Summary

Aether implements the first software layer for the GreenServe research plan:
modeling and measuring intelligence per watt for LLM serving systems.

The relevant system components are:

- context-length-aware routing pools inspired by FleetOpt
- KV-cache compression and its effect on in-flight capacity
- compute-bound prefill and memory-bound decode phase asymmetry
- recompute-vs-swap scheduling decisions under VRAM pressure
- optional SGLang metrics collection for future real experiments

The initial implementation is deliberately simulation-first. It lets the team
debug formulas, scenario schemas, CLI behavior, and normalized output formats on
CPU before running expensive GPU experiments.
