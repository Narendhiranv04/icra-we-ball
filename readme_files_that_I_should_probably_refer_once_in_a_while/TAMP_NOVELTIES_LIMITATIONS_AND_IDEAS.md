# TAMP paper novelties, limitations, extensions, and research ideas

**Coverage cutoff:** 2026-07-14  
**Strict/core analyses:** 216, one for every bibliography record.  
**Separate precursor analyses:** 9.  
**Bibliography and ordering:** [TAMP_PAPERS_LATEST_FIRST.md](./TAMP_PAPERS_LATEST_FIRST.md)

## How to read this analysis

Every entry uses the same stable ID as the bibliography. **Novelty** reports the paper's central TAMP contribution. **Limitations** are labeled *explicit* only when the paper or its reported evaluation directly establishes them; otherwise they are labeled *inferred* and are conservative deductions from the assumptions, method, or evaluation scope. **Extensions / research idea** is a researcher-level next step, not necessarily a claim made by the authors.

This distinction matters: absence of an experiment is evidence of an untested scope, not proof that a method cannot work there.

## Cross-cutting research gaps

The per-paper notes below repeatedly expose a small set of durable open problems:

1. **Open-world, partially observable TAMP with calibrated uncertainty.** Perception, language models, geometric models, and action outcomes all need uncertainty estimates that survive composition over long horizons.
2. **Automatic but verifiable domain construction.** Learned predicates, operators, samplers, constraints, and skills reduce engineering effort, but need counterexample-driven repair and guarantees against silently deleting feasible solutions.
3. **Dynamics- and contact-aware long-horizon planning.** Most scalable systems still rely on quasi-static pick/place abstractions; contact-rich, deformable, non-prehensile, and whole-body tasks remain difficult.
4. **Temporal and multi-robot TAMP.** Concurrent action, uncertain duration, shared resources, asynchronous execution, and online rescheduling are still much less mature than sequential single-robot planning.
5. **Principled learned guidance.** Learned feasibility, value, decomposition, and language guidance should accelerate search without destroying completeness, soundness, risk calibration, or out-of-distribution robustness.
6. **Execution as part of planning.** Monitoring, active perception, failure diagnosis, recovery, and closed-loop skills should be first-class planning operators rather than a downstream patch.
7. **Benchmarks and reproducibility.** The field still lacks a single representation, benchmark suite, and metric set spanning symbolic difficulty, geometric narrow passages, dynamics, uncertainty, execution failure, and real-robot transfer.
8. **Continual and anticipatory TAMP.** Planners should learn across tasks, preserve useful world arrangements, reason about future unknown requests, and avoid catastrophic model drift.

## Researcher synthesis: how the field changed

- **1989-2004 — multimodal foundation:** manipulation was recast as transitions among grasp, placement, transit, and transfer manifolds. The enduring limitation was specialized geometry with no general task language.
- **2004-2010 — hybrid interfaces:** aSyMov, semantic attachments, and coupled symbolic/geometric search made geometry participate during task search. Hand-engineered interfaces and costly black-box calls became the central bottleneck.
- **2010-2013 — hierarchy and interleaving:** continual refinement, suggesters, caching, interval constraints, and cross-level backtracking reduced premature full grounding, but correctness depended strongly on domain hierarchies and finite-sample assumptions.
- **2013-2016 — constraints, explanations, and optimization:** SMT, nonlinear programming, relaxations, culprit sets, and learned heuristics turned geometric failure into search guidance. Nonconvex local failure and weak/expensive explanations remained hard to distinguish from true infeasibility.
- **2017-2022 — guarantees, learning, and uncertainty:** streams, factored sampling, asymptotic analysis, learned predicates/samplers/values, temporal logic, stochastic policies, and multi-robot formulations substantially broadened TAMP. Distribution shift and the gap between nominal planning and execution became more visible.
- **2023-2026 — foundation models, GPUs, dynamics, and concurrency:** LLM/VLM guidance, differentiable and GPU-parallel optimization, active perception, metric time, asynchronous teams, and contact/whole-body systems now dominate the frontier. The research opportunity is not simply more learned guidance, but **calibrated guidance wrapped in verifiable, completeness-preserving planning and closed-loop execution**.

## Per-paper analysis

### T001 — HTNs, Streams, and Common-Sense Models

- **Novelty:** Integrates totally ordered HTNs, PDDLStream-style conditional generators, geometric world-model predicates, and failure-guided backtracking for dual-arm kitchen manipulation.
- **Limitations:** *Explicit:* the method is closed-world, fully observable, deterministic, and totally ordered; generator failure dominates runtime, while causal backtracking can miss world-state effects.
- **Extension / idea:** Add partial-order online execution, uncertainty, cached geometric subproblems, and causal failure explanations that include state-changing side effects.

### T002 — Multi-Agent TAMP Trends Survey

- **Novelty:** Systematically classifies 60 multi-agent TAMP works from 2021-2025 along task allocation, continuous coordination, communication, uncertainty, and emerging learning dimensions.
- **Limitations:** *Inherent review limitation:* the five-year, multi-agent-only window omits the classical foundations and cannot cover work after its search cutoff.
- **Extension / idea:** Maintain a living evidence map with benchmark scale, concurrency model, uncertainty type, guarantees, and reproducibility artifacts for every system.

### T003 — Optimizing Trajectory-Trees in Belief Space

- **Novelty:** Extends logic-geometric planning to branching belief-space trajectory trees and solves the coupled branches with distributed augmented-Lagrangian optimization.
- **Limitations:** *Explicit:* experiments use small belief trees; branching and nonlinear trajectory optimization grow rapidly.
- **Extension / idea:** Learn macro-actions and branch-value bounds, then compress or expand beliefs adaptively.

### T004 — VAP-TAMP

- **Novelty:** Makes active perception a task-planning decision, using action knowledge, visual-language reasoning, and a scene graph to handle missing situation information.
- **Limitations:** *Inferred:* correctness inherits VLM and perception errors, while the demonstrated service-task scope does not establish generality.
- **Extension / idea:** Represent calibrated object-belief distributions and plan information-gathering actions by expected downstream feasibility gain.

### T005 — Graph-of-Constraints MPC

- **Novelty:** Represents a multi-agent task as a partially ordered graph of geometric constraints and couples it to MPC for reactive reassignment and motion.
- **Limitations:** *Inferred:* performance depends on a correct constraint graph and reliable 3-D tracking.
- **Extension / idea:** Induce constraint graphs from demonstrations and propagate perception uncertainty through reassignment decisions.

### T006 — Interleaving Scheduling and Motion Planning

- **Novelty:** Alternates a scheduler with motion validation and incrementally learns symbolic space-time conflicts and duration abstractions.
- **Limitations:** *Inferred:* predefined task models and repeated motion checks remain costly; dynamic arrivals and uncertain durations are not central.
- **Extension / idea:** Learn probabilistic conflict explanations and support online rescheduling with nonstationary action times.

### T007 — CoCo-TAMP

- **Novelty:** Uses LLM commonsense as a prior for hidden-state estimation inside partially observable TAMP.
- **Limitations:** *Inferred:* biased or confidently wrong language priors can corrupt long-horizon belief updates.
- **Extension / idea:** Calibrate the language prior against observations and trigger active perception when prior and geometric evidence disagree.

### T008 — HAD-TAMP

- **Novelty:** Combines receding-horizon PDDL TAMP with a context reasoner that selects human-aware and ergonomic motion planners during human intervention.
- **Limitations:** *Explicit evaluation scope:* one industrial draping cell and a small trial set do not establish transfer across users and workflows.
- **Extension / idea:** Maintain a probabilistic belief over human intent and workload, with certified safety-aware replanning across tasks.

### T009 — Lang2Manip

- **Novelty:** Connects LLM-generated symbolic plans to the Kautham geometric-planning stack in a robot-agnostic manipulation toolchain.
- **Limitations:** *Inferred:* syntactic correctness and geometric calls alone do not guarantee safe, closed-loop execution.
- **Extension / idea:** Add typed plan validation, formal interface contracts, and perception-driven execution repair.

### T010 — Cross-Entropy Optimization of Physically Grounded Plans

- **Novelty:** Uses GPU physics and cross-entropy optimization to select hybrid task/controller decisions by physical outcome rather than only kinematic feasibility.
- **Limitations:** *Inferred:* compute cost, simulator mismatch, and stochastic optimization preclude a completeness guarantee.
- **Extension / idea:** Couple system identification and risk-sensitive objectives to sim-to-real validation.

### T011 — High-Performance Dual-Arm TAMP

- **Novelty:** Couples dependency-aware task planning with massively parallel GPU motion planning for synchronized two-arm rearrangement.
- **Limitations:** *Explicit evaluation scope:* fixed dual-arm tabletop settings leave heterogeneous teams and changing scenes untested.
- **Extension / idea:** Generalize the dependency representation to mobile manipulators, uncertain durations, and online insertion of new tasks.

### T012 — ScheduleStream

- **Novelty:** Extends stream-based TAMP with durative parameterized actions, asynchronous multi-arm schedules, and GPU samplers.
- **Limitations:** *Inferred:* domains still require action-duration models and sampler interfaces; duration uncertainty is not resolved by nominal scheduling.
- **Extension / idea:** Integrate resource constraints, temporal uncertainty, and receding-horizon rescheduling.

### T013 — Kinodynamic VLM-Guided TAMP

- **Novelty:** Interleaves a hybrid kinodynamic search tree with VLM guidance based on rendered candidate states.
- **Limitations:** *Inferred:* rendering fidelity and VLM false pruning can hide feasible branches, while simulation-to-real dynamics remain uncertain.
- **Extension / idea:** Use confidence-aware, reversible guidance and retain a completeness-preserving unguided sampling budget.

### T014 — VIZ-COAST

- **Novelty:** Converts VLM explanations of downward-refinement failures into constraints that prune later TAMP search.
- **Limitations:** *Inferred:* an incorrect explanation may permanently eliminate feasible solutions.
- **Extension / idea:** Make learned constraints retractable, attach confidence and counterexamples, and periodically audit pruned branches.

### T015 — Systematic Study of LLMs with PDDLStream

- **Novelty:** Provides a controlled study of 16 LLM substitutions over 4,950 TAMP problem instances rather than only a new prompt pipeline.
- **Limitations:** *Explicit evaluation scope:* one model family, zero-shot use, three domains, and PDDLStream-specific interfaces constrain generalization.
- **Extension / idea:** Compare multiple model families, fine-tuning regimes, representations, and completeness-preserving hybrid policies.

### T016 — Tighter Convex Relaxation with Logic Network Flow

- **Novelty:** Strengthens mixed-integer TAMP relaxations by encoding logical transitions as a network flow rather than a weak big-M structure.
- **Limitations:** *Inferred:* the strongest formulation relies on polyhedral or piecewise-affine structure and still faces combinatorial growth.
- **Extension / idea:** Build certified relaxations for nonlinear contact dynamics and learn cut selection without weakening correctness.

### T017 — Reactive Multi-Robot EV-Battery Disassembly

- **Novelty:** Closes the loop between task planning, dual-robot motion, and reactive collision handling for a physically consequential disassembly domain.
- **Limitations:** *Explicit evaluation scope:* one product family and robot cell do not establish transfer across layouts or damage states.
- **Extension / idea:** Add uncertain perception of fasteners/materials and automatic domain adaptation to new product variants.

### T018 — HyperSTL TAMP

- **Novelty:** Plans over sets of trajectories so hyperproperties can express relationships among multiple dynamic-system executions.
- **Limitations:** *Inferred:* alternating trajectory quantifiers and discrete-time encodings create severe scaling pressure.
- **Extension / idea:** Develop compositional, stochastic, and decentralized HyperSTL abstractions with continuous-time certificates.

### T019 — Expanding AND/OR Graphs

- **Novelty:** Expands alternative task/motion refinements online instead of committing to a fully enumerated AND/OR model.
- **Limitations:** *Explicit evaluation scope:* two benchmarks and simulated mobile-robot cases leave model construction and hardware robustness open.
- **Extension / idea:** Learn graph expansions and their feasibility priors from execution traces while preserving fallback branches.

### T020 — Benders Decomposition for STL Bipedal TAMP

- **Novelty:** Separates discrete STL task choices from expensive biped dynamics and feeds subproblem information back through Benders cuts.
- **Limitations:** *Inferred:* convergence and cut quality depend on the chosen decomposition and dynamics approximation.
- **Extension / idea:** Learn warm starts and reusable cuts, then test online replanning with model mismatch.

### T021 — Humanoid Loco-Manipulation TAMP

- **Novelty:** Integrates symbolic actions, contact choices, locomotion, manipulation, and whole-body dynamics in one long-horizon formulation.
- **Limitations:** *Inferred:* whole-body optimization is computationally heavy and sensitive to contact/model fidelity.
- **Extension / idea:** Combine hierarchical contact abstractions with fast reactive policies and online state-estimation feedback.

### T022 — Partially Observable Household TAMP

- **Novelty:** Adds task-level backward relevance search and an object-manipulation constraint graph to PDDLStream so a household robot explores and moves only objects implicated in a partially observed task.
- **Limitations:** *Inferred:* relevance reasoning can miss hidden or indirectly necessary objects, and the constraint graph depends on accurate object-state estimates.
- **Extension / idea:** Maintain a belief over hidden dependencies and choose views or exploratory manipulations by expected plan-feasibility gain.

### T023 — Infinite Completion Trees with Agnostic Skills

- **Novelty:** Combines STAP-style pretrained task-agnostic policies, effort-level search over an infinite completion tree, and a learned success estimator for long-horizon skill sequencing.
- **Limitations:** *Inferred:* reliability inherits policy and estimator calibration, while empirical speedups do not certify behavior under novel objects or dynamics.
- **Extension / idea:** Learn uncertainty-aware skill outcomes online and preserve a systematic effort-allocation fallback when the estimator is out of distribution.

### T024 — Language-Grounded Multi-Robot 3-D Scene Graphs

- **Novelty:** Shares hierarchical 3-D scene graphs across robots for language grounding, allocation, motion planning, and execution.
- **Limitations:** *Inferred:* map inconsistency, perception errors, and LLM goal errors can propagate across the whole team.
- **Extension / idea:** Maintain distributed probabilistic maps and reallocate tasks when communication or localization degrades.

### T025 — Prime the Search

- **Novelty:** Derives geometry-relevant predicates from motion reasoning and uses an LLM to warm-start Monte Carlo tree search.
- **Limitations:** *Explicit evaluation scope:* six problem families and engineered predicate interfaces leave out-of-distribution reliability unresolved.
- **Extension / idea:** Learn abstractions with uncertainty and reserve a guaranteed exploration budget independent of the LLM prior.

### T026 — ViLaIn-TAMP

- **Novelty:** Translates visual-language scene understanding into formal TAMP specifications and uses planning failures as refinement feedback.
- **Limitations:** *Inferred:* the interpreter has no formal semantic-correctness guarantee and is evaluated in a narrow cooking-style domain.
- **Extension / idea:** Verify generated predicates against perception queries and use minimal counterexamples for interactive repair.

### T027 — One Demo Is All It Takes

- **Novelty:** Induces predicates and action models from one demonstration by combining LLM proposals with simulated rollouts.
- **Limitations:** *Inferred:* rare preconditions, incomplete demonstrations, and simulator errors can yield unsound domains.
- **Extension / idea:** Actively generate counterexample rollouts and quantify posterior uncertainty over alternative domains.

### T028 — Meta-Optimization and Program Search

- **Novelty:** Lets an LM search over trajectory-optimization programs while a zero-order outer loop tunes continuous parameters.
- **Limitations:** *Inferred:* many generated programs are invalid or costly to evaluate, and no global guarantee follows.
- **Extension / idea:** Constrain generation to a typed, verified DSL and add differentiable repair plus cached subprogram evaluation.

### T029 — LLM-PAS

- **Novelty:** Uses refined LLM prompting to leave selected constraints partially grounded until online execution and to repair PDDL goals.
- **Limitations:** *Inferred:* invalid partial plans may be unsafe and language-generated repairs are not correctness guarantees.
- **Extension / idea:** Add runtime safety shields, executable monitors, and counterexample-guided prompt/domain repair.

### T030 — Hierarchical Temporal-Logic Multi-Robot TAMP

- **Novelty:** Combines hierarchical temporal-logic allocation with graph-of-convex-sets motion planning in a product construction with theoretical properties.
- **Limitations:** *Inferred:* product-state growth and convex-set assumptions restrict large, nonconvex, dynamic teams.
- **Extension / idea:** Build decentralized and incremental product graphs with learned but certified decompositions.

### T031 — Temporal TAMP with Metric Time

- **Novelty:** Formalizes temporal TAMP for multiple moving objects and interleaves incremental SMT with duration/deadline-aware motion planning and contextualized conflict learning.
- **Limitations:** *Inferred:* deterministic metric durations and known obstacles understate execution-time and motion uncertainty; SMT conflict sets can still grow sharply.
- **Extension / idea:** Add probabilistic duration bounds, asynchronous execution, and receding-horizon invalidation of learned conflicts.

### T032 — Lazy-DaSH

- **Novelty:** Delays validation of expensive composite spaces in a hypergraph-based multi-robot TAMP search until they are selected.
- **Limitations:** *Inferred:* laziness can discover decisive geometric failure late; evaluation covers four scenarios.
- **Extension / idea:** Learn validation order and reuse conflict certificates during dynamic replanning.

### T033 — APEX-MR

- **Novelty:** Post-processes task-motion plans into asynchronous, collision-safe multi-robot assembly execution.
- **Limitations:** *Inferred:* it assumes supplied plans and is demonstrated mainly on dual-arm LEGO-like assembly.
- **Extension / idea:** Integrate planning, resource reasoning, online schedule repair, and failure recovery in one loop.

### T034 — Iteratively Deepened AND/OR Graph Networks

- **Novelty:** Uses iterative deepening over AND/OR alternatives to handle unknown task refinements and coordinate multiple robots.
- **Limitations:** *Inferred:* demonstrations assume known object poses and engineered alternatives.
- **Extension / idea:** Infer graph structure from observation and expand it only where belief or execution failures demand.

### T035 — BLISS

- **Novelty:** Formulates snake-robot TAMP under localization failure as a chance-constrained hybrid partially observable optimization problem.
- **Limitations:** *Inferred:* convex/linearized models and a specialized platform limit transfer to general nonlinear contacts.
- **Extension / idea:** Add nonlinear belief dynamics, adaptive sensing actions, and risk allocation learned across missions.

### T036 — Onto-LLM-TAMP

- **Novelty:** Enriches LLM prompts with an ontology so semantic, task, robot, and spatial knowledge constrain symbolic-to-geometric planning.
- **Limitations:** *Inferred:* ontology population and prompt design remain engineered, and the demonstrated hierarchical placement scope is narrow.
- **Extension / idea:** Learn ontology updates from execution while using a reasoner to reject inconsistent LLM additions.

### T037 — Differentiable GPU-Parallelized TAMP

- **Novelty:** Batches many task skeleton seeds on a GPU and differentiates through continuous optimization to accelerate joint refinement.
- **Limitations:** *Inferred:* it requires differentiable models, remains vulnerable to local optima, and does not make discrete search complete by itself.
- **Extension / idea:** Add nonsmooth contact models, adaptive skeleton generation, and a completeness-preserving global-search wrapper.

### T038 — OWL-TAMP

- **Novelty:** Lets a VLM generate both symbolic ordering constraints and executable continuous constraint code for previously unseen open-world tasks.
- **Limitations:** *Inferred:* generated code can be wrong or unsafe and has no semantic or completeness guarantee.
- **Extension / idea:** Sandbox and formally check constraint code, attach confidence, and retract constraints after counterexamples.

### T039 — CaStL

- **Novelty:** Uses an LLM to translate natural-language goals into reusable constraint specifications, then grounds and verifies them through a full long-horizon task-and-motion stack.
- **Limitations:** *Inferred:* language-to-constraint translation can be semantically wrong or omit safety conditions, and geometric verification cannot detect every intent error.
- **Extension / idea:** Type-check the constraint language, ask targeted clarification on ambiguity, and synthesize minimal counterexamples when planning fails.

### T040 — SkillMimicGen / SkillGen

- **Novelty:** Segments a few contact-rich demonstrations, adapts them to new contexts, and stitches them with free-space TAMP to mass-produce training data and deploy hybrid skills.
- **Limitations:** *Explicit evaluation scope:* it begins from segmented demonstrations and tests three real tasks; task-order discovery is not the main problem solved.
- **Extension / idea:** Jointly infer segmentation, skill preconditions, and alternative task skeletons from failed executions.

### T041 — Automated Planning Domain Inference

- **Novelty:** Infers and iteratively refines TAMP action preconditions and effects from a small number of continuous robot trajectories.
- **Limitations:** *Inferred:* missing observations can create incomplete or unsound operators, especially for rare geometric conditions.
- **Extension / idea:** Select informative experiments and use formal counterexamples to maintain a versioned set of candidate domains.

### T042 — Deadline-Aware Effort Allocation

- **Novelty:** Treats how long to spend on alternative motion-refinement attempts as an MDP/metareasoning problem under a deadline.
- **Limitations:** *Explicit/theoretical:* the allocation problem is hard and relies on candidate timing distributions that may be misspecified.
- **Extension / idea:** Learn nonstationary solve-time models online and jointly decide which skeletons and refinements to generate.

### T043 — VLM-Guided Long-Horizon TAMP

- **Novelty:** Asks a VLM for intermediate subgoals while retaining TAMP as an executable geometric verifier.
- **Limitations:** *Explicit evaluation scope:* kitchen-style domains and proprietary model queries do not show that proposed subgoals preserve solvability.
- **Extension / idea:** Check subgoal admissibility, cache verified decompositions, and fall back to unbiased search on low confidence.

### T044 — Logic-Network-Flow STL TAMP

- **Novelty:** Places temporal-logic predicates on network-flow transitions to tighten a mixed-integer task-motion formulation.
- **Limitations:** *Inferred:* branch growth remains combinatorial and the formulation is strongest for polyhedral, piecewise-affine systems.
- **Extension / idea:** Compose local flows and certified nonlinear relaxations for contact-rich dynamics.

### T045 — Interpretable Responsibility Sharing

- **Novelty:** Introduces auxiliary objects and object-responsibility scores as an interpretable heuristic for redistributing manipulation burden.
- **Limitations:** *Inferred:* affordances and sharing rules are engineered and tested in limited household-style settings.
- **Extension / idea:** Learn transferable responsibility models with uncertainty and expose their causal evidence to the planner.

### T046 — Learning Long-Horizon Action Dependencies

- **Novelty:** Learns long-range action-sample dependencies with a transformer so bilevel search can reject or prioritize task-motion refinements before expensive optimization.
- **Limitations:** *Explicit evaluation scope:* five quasi-static simulation domains leave out-of-distribution calibration and hardware dynamics unresolved.
- **Extension / idea:** Use conformal feasibility scores and a completeness-preserving fallback while adapting from real execution traces.

### T047 — Meta-Engine

- **Novelty:** Adds topology-driven refinements around an otherwise planner-independent interleaved TAMP engine.
- **Limitations:** *Inferred:* the benefit depends on topology extraction and its ability to represent moving agents and obstacles.
- **Extension / idea:** Learn which refinement to invoke and update topological classes incrementally under dynamics.

### T048 — Practical Finite Sample Bounds

- **Novelty:** Derives high-probability upper bounds on the motion samples needed by sampled TAMP and supplies numerically tighter practical estimates.
- **Limitations:** *Explicit/theoretical:* bounds can become loose or enormous in high-dimensional/narrow geometry.
- **Extension / idea:** Derive instance-adaptive, geometry-aware bounds and use them to stop or redirect refinement online.

### T049 — Anticipatory TAMP

- **Novelty:** Learns a future-task cost and reranks current feasible plans so the final scene is useful for likely later requests.
- **Limitations:** *Inferred:* utility depends on the training distribution of future tasks and on persistent-world assumptions.
- **Extension / idea:** Detect distribution shift, support multiple users, and plan information-preserving arrangements under uncertain future demand.

### T050 — Grasping with Rearrangement in Confined Workspaces

- **Novelty:** Searches over prehensile and non-prehensile rearrangement choices with motion-feasibility checks to expose and grasp targets in confined clutter, including real-robot validation.
- **Limitations:** *Inferred:* known rigid-object models and a confined-workspace manipulation setup leave deformability, perception uncertainty, and stochastic contact open.
- **Extension / idea:** Couple belief-space object tracking to risk-aware pushes, learned contact outcomes, and recovery after an unexpected displacement.

### T051 — TAMPER

- **Novelty:** Combines partial grounding, closed-loop behaviors, execution monitoring, and learned failure constraints to make nominal TAMP plans executable in reality.
- **Limitations:** *Explicit evaluation scope:* predefined behaviors and roughly forty physical trials do not cover open-ended failures.
- **Extension / idea:** Discover repair behaviors and probabilistic failure explanations continually from fleet execution.

### T052 — COAST

- **Novelty:** Couples constraints and sampling streams in a focused interleaving scheme with a probabilistic-completeness argument.
- **Limitations:** *Explicit evaluation scope:* three domains and hand-authored stream/constraint interfaces leave engineering and dynamic-scene costs open.
- **Extension / idea:** Learn streams and constraint explanations while retaining a certified fallback sampler.

### T053 — Effort-Level Search in Infinite Completion Trees

- **Novelty:** Models hierarchical TAMP computation as an infinite completion tree with unknown preemptible effort and uses polynomial level sets to balance depth, branching, and random restarts while retaining completeness.
- **Limitations:** *Inferred:* its model assumes useful preemption and synthetic/tree benchmarks cannot fully predict expensive real solver-state reuse.
- **Extension / idea:** Learn Bayesian solve-time and restart models online, include memory/parallelism costs, and report anytime probability-of-completion bounds.

### T054 — Optimistic RL Skill Insertions

- **Novelty:** Gives learned skills symbolic interfaces and inserts them optimistically into task plans before continuous refinement.
- **Limitations:** *Inferred:* skills are trained separately and interface errors can make a symbolic plan misleading.
- **Extension / idea:** Discover skills continually, learn stochastic preconditions/effects, and retract unsafe insertions from counterexamples.

### T055 — Optimization-Based TAMP Survey

- **Novelty:** Organizes classical and learning-based optimization TAMP by formulation, decomposition, solver, and application.
- **Limitations:** *Inherent review limitation:* it is a time-bounded taxonomy and predates the 2025-2026 VLM, temporal, and GPU wave.
- **Extension / idea:** Maintain a living benchmark-linked taxonomy that records guarantees, model assumptions, and reproducibility artifacts.

### T056 — Sketch Decompositions

- **Novelty:** Uses domain sketches to decompose combined TAMP into width-one subproblems with local retry structure.
- **Limitations:** *Inferred:* performance and guarantees depend on a suitable engineered sketch and its structural assumptions.
- **Extension / idea:** Learn sketches from successful and failed plans, then verify their width and dead-end properties.

### T057 — LLM3

- **Novelty:** Uses an LLM for task selection and continuous parameters, then reasons from motion-planning failures to revise the plan.
- **Limitations:** *Explicit evaluation scope:* simulated box-style tasks and qualitative robot tests leave hallucination, query cost, and broad transfer unresolved.
- **Extension / idea:** Convert failures to formal minimal conflicts and restrict language outputs to a typed action/parameter grammar.

### T058 — TAMPURA

- **Novelty:** Learns an abstract stochastic model over closed-loop controllers and plans for information, uncertainty, and risk under partial observability.
- **Limitations:** *Inferred:* supplied controllers and preconditions bound the reachable behavior set, while rare outcomes are hard to learn.
- **Extension / idea:** Discover controllers and abstractions online with calibrated Bayesian uncertainty and risk-sensitive exploration.

### T059 — Hierarchical 3-D Scene Graph TAMP

- **Novelty:** Plans over sparse object instances and reveals geometric detail only for currently relevant parts of a hierarchical 3-D scene graph.
- **Limitations:** *Explicit evaluation scope:* two graphs and largely static scenes leave graph errors and dynamics untested.
- **Extension / idea:** Add probabilistic graph maintenance, active object discovery, and execution-triggered refinement.

### T060 — Object-Centric Motion-Constraint Abstractions

- **Novelty:** Unifies task and motion heuristic search through object-centric abstractions derived from continuous motion constraints.
- **Limitations:** *Inferred:* conservative abstractions can suppress useful interactions and assume known object/motion models.
- **Extension / idea:** Learn abstraction granularity online and represent uncertainty about constraint relevance.

### T061 — D-LGP

- **Novelty:** Dynamically grows a logic-geometric search tree and re-optimizes globally to react to environmental change.
- **Limitations:** *Explicit evaluation scope:* three benchmarks do not remove sensitivity to models, compute budgets, and local trajectory optima.
- **Extension / idea:** Add chance constraints, contact uncertainty, and reusable warm starts for hard real-time reactions.

### T062 — Multi-Modal MPPI and Active Inference

- **Novelty:** Blends sampling-based predictive control, multiple skills, and active-inference objectives for reactive hybrid task/motion choice.
- **Limitations:** *Inferred:* computational load and physics-model mismatch can destabilize long-horizon decisions.
- **Extension / idea:** Learn uncertainty-aware dynamics and prune modes with risk bounds rather than point predictions.

### T063 — Offline Skill Generalization through TAMP

- **Novelty:** Uses planner-generated experience to improve an offline-RL skill that is then called again by the planner.
- **Limitations:** *Explicit evaluation scope:* scripted predicates and simulated block pushing test one skill family.
- **Extension / idea:** Co-learn several stochastic skill interfaces on hardware and propagate epistemic uncertainty into planning.

### T064 — Multi-Robot Feasibility Prediction for Realistic Objects

- **Novelty:** Extends learned TAMP feasibility guidance from box-like single/dual-robot cases to mesh-shaped objects and collaborative multi-robot manipulation.
- **Limitations:** *Inferred:* supervised predictions inherit the training distribution and false negatives can suppress feasible task branches.
- **Extension / idea:** Calibrate per-robot and joint-feasibility uncertainty, audit rejected branches, and retain a completeness-preserving unguided refinement channel.

### T065 — NOD-TAMP

- **Novelty:** Uses neural object descriptors to adapt contact-rich skill trajectories to new objects and composes those skills through TAMP.
- **Limitations:** *Inferred:* descriptor failures under topology changes, occlusion, or unseen affordances can silently invalidate refinements.
- **Extension / idea:** Attach descriptor uncertainty, request active views, and adapt from corrective demonstrations.

### T066 — PROTAMP-RRT

- **Novelty:** Grows symbolic and geometric choices in one RRT-like structure and uses probabilistic information to redirect refinement and replanning.
- **Limitations:** *Explicit evaluation scope:* simulated pick-and-place tests leave high-DOF hardware, uncertainty, and formal guarantees open.
- **Extension / idea:** Preserve probabilistic completeness with adaptive bias and validate on perception-driven real manipulation.

### T067 — Human-in-the-Loop TAMP for Imitation Learning

- **Novelty:** Lets TAMP solve modelled free-space structure and asks a human only for demonstrations of contact-rich gaps.
- **Limitations:** *Inferred:* it still relies on a hand-built planning model and a human who can recognize and solve missing segments.
- **Extension / idea:** Learn when intervention is valuable and convert repeated interventions into reusable operators automatically.

### T068 — Modular Multi-Level Replanning

- **Novelty:** Separates logical and motion repair into a modular stack that can replan during real stacking and rearrangement.
- **Limitations:** *Inferred:* nominal domain models and limited dynamic cases do not establish broad failure coverage or guarantees.
- **Extension / idea:** Learn cross-level failure diagnoses and maintain beliefs over moving/occluded objects.

### T069 — Multi-Robot Geometric TAMP

- **Novelty:** Couples reachability/occlusion graphs, mixed-integer allocation, and Monte Carlo tree search for collaborative manipulation.
- **Limitations:** *Explicit evaluation scope:* synchronous, mostly monotone problems with roughly five to six clutter objects expose scaling limits.
- **Extension / idea:** Support asynchronous nonmonotone rearrangement and learn when layout changes are worth the motion cost.

### T070 — Experience-Based TAMP on Foliated Manifolds

- **Novelty:** Reuses experience as a roadmap whose nodes and edges reflect foliated constraint manifolds across related tasks.
- **Limitations:** *Inferred:* transfer depends on manifold correspondence and is shown in limited real settings.
- **Extension / idea:** Adaptively validate transferred experience and retain an unbiased sampler for novel topology.

### T071 — R-LGP

- **Novelty:** Uses a reachability graph to guide and prune logic-geometric programs for mobile manipulation.
- **Limitations:** *Inferred:* graph construction, discretization, and local trajectory optimization can dominate cost or miss feasible corridors.
- **Extension / idea:** Maintain the graph incrementally and refine resolution only near uncertain feasibility boundaries.

### T072 — STAMP

- **Novelty:** Applies Stein variational gradient descent through differentiable physics to maintain diverse joint task-motion candidates.
- **Limitations:** *Inferred:* differentiability breaks at many contacts, and diversity is not a completeness or safety certificate.
- **Extension / idea:** Combine verified hybrid contact modes, nondifferentiable repair, and completeness-preserving exploration.

### T073 — Asynchronous Task-Plan Refinement

- **Novelty:** Refines a given multi-robot task plan asynchronously with hierarchical CSPs and implicit-time motion roadmaps.
- **Limitations:** *Inferred:* assuming an input task plan separates refinement from discovery of better task structures.
- **Extension / idea:** Interleave task search, asynchronous execution, and duration/resource uncertainty.

### T074 — Learning Reusable Manipulation Strategies

- **Novelty:** Learns reusable contact-mode mechanisms and their continuous samplers from one demonstration plus self-play, then recomposes them with a general TAMP planner.
- **Limitations:** *Inferred:* it assumes recognizable contact structure and a sufficiently faithful simulator.
- **Extension / idea:** Induce mechanisms with calibrated uncertainty, active counterexamples, and online real-world repair.

### T075 — S3O

- **Novelty:** Jointly optimizes abstract navigation locations and geometric grounding to reduce long-horizon mobile-manipulation cost.
- **Limitations:** *Explicit evaluation scope:* static, largely two-dimensional navigation abstractions limit richer 3-D scenes.
- **Extension / idea:** Optimize belief-dependent 3-D base/whole-body configurations online.

### T076 — Embodied Lifelong TAMP

- **Novelty:** Maintains a lifelong mixture of continuous samplers and improves them across a stream of related task instances.
- **Limitations:** *Explicit evaluation scope:* 2-D and BEHAVIOR-style experiments focus on samplers rather than full model drift.
- **Extension / idea:** Handle real-robot domain shift, catastrophic forgetting, and continual operator as well as sampler learning.

### T077 — Recent Trends in TAMP Survey

- **Novelty:** Synthesizes integrated, hierarchical, sampling, optimization, and learning TAMP trends through the early 2020s.
- **Limitations:** *Inherent review limitation:* later VLM, differentiable-GPU, temporal, and open-world work is outside its cutoff.
- **Extension / idea:** Link a living taxonomy to executable benchmark instances and versioned evidence tables.

### T078 — Deep-Vision Spatial Reasoning

- **Novelty:** Predicts which objects matter geometrically so a sequential-manipulation planner can focus its search.
- **Limitations:** *Inferred:* vision-model distribution shift can misclassify a necessary object, with no feasibility certificate.
- **Extension / idea:** Use conformal uncertainty and retain a complete fallback ordering over omitted objects.

### T079 — DiMSam

- **Novelty:** Learns diffusion-based continuous samplers for PDDLStream-style TAMP from partial point-cloud observations.
- **Limitations:** *Explicit evaluation scope:* simulation and sparse observations leave collision, slip, state-estimation, and real transfer errors open.
- **Extension / idea:** Train calibrated conditional samplers and close the loop with active perception and outcome feedback.

### T080 — AutoTAMP

- **Novelty:** Uses LLMs as formal-specification translators and checkers in an iterative plan-repair loop.
- **Limitations:** *Inferred:* prompt vocabulary bounds coverage and repeated language correction has no soundness or convergence guarantee.
- **Extension / idea:** Type-check translations, extract proof obligations, and repair with minimal formal counterexamples.

### T081 — Interleaved Integrated TAMP

- **Novelty:** Interleaves depth-first task search with Monte Carlo motion refinement and analyzes completeness/optimal behavior under its assumptions.
- **Limitations:** *Inferred:* simulation evidence and accurate feasibility/cost models leave large-scale uncertain execution unresolved.
- **Extension / idea:** Learn refinement costs online and allocate sampling with anytime bounds.

### T082 — Process-Aware Source-Seeking TAMP

- **Novelty:** Couples discrete process-aware source-seeking actions to continuous robot motion and extends the loop with dynamic event-triggered replanning.
- **Limitations:** *Inferred:* success depends on the assumed source/process model and trigger design; the application scope does not establish broad manipulation transfer.
- **Extension / idea:** Plan over a learned field belief, adapt trigger thresholds to value of information, and coordinate multiple sensing robots under communication limits.

### T083 — Video-Guided Multi-Contact TAMP

- **Novelty:** Extracts contact and pose structure from video, connects subtasks with multiple RRTs, and in the expanded version adds trajectory refinement/optimal control and broader generalization tests.
- **Limitations:** *Explicit evaluation scope:* noisy video extraction and three principal benchmark tasks leave category-level transfer open.
- **Extension / idea:** Use uncertainty-aware video parsing and retrieve reusable contact templates from large video collections.

### T084 — OPTIMUS

- **Novelty:** Uses TAMP to generate supervision for a visuomotor transformer that amortizes long-horizon manipulation planning.
- **Limitations:** *Explicit:* reported performance is imperfect and still depends on a planner-generated supervisor and model transfer.
- **Extension / idea:** Detect policy uncertainty and call TAMP selectively for closed-loop correction and new data.

### T085 — Differentiable Task Assignment and Motion Planning

- **Novelty:** Relaxes task assignment and implicit action choices into a fully differentiable nonlinear program with motion.
- **Limitations:** *Explicit evaluation scope:* simulated pick-and-place problems remain nonconvex and provide no discrete global-optimality guarantee.
- **Extension / idea:** Add contact-rich dynamics and branch-and-bound certificates around differentiable relaxations.

### T086 — Human-Robot Multi-Agent TAMP and Execution

- **Novelty:** Couples timeline-based symbolic planning with online motion, duration, and assignment decisions for human-robot workcells.
- **Limitations:** *Explicit evaluation scope:* one mosaic/manufacturing setting and fixed human-duration models limit transfer.
- **Extension / idea:** Learn human intent and duration uncertainty while preserving explainable temporal safety constraints.

### T087 — Optimal Grasps and Placements in Clutter

- **Novelty:** Optimizes continuous grasp and placement parameters under clutter geometry and feeds better grounding information back into task search.
- **Limitations:** *Explicit evaluation scope:* simulated clutter and local/model-based optimization leave perception error, contact dynamics, and global optimality open.
- **Extension / idea:** Optimize over belief distributions, return certified lower bounds or diverse alternatives, and update grasps from execution feedback.

### T088 — Simultaneous Action and Grasp Feasibility Prediction

- **Novelty:** Uses one multi-task network to predict both symbolic-action feasibility and viable grasp types, giving the task planner more geometric guidance than a binary action score.
- **Limitations:** *Inferred:* fully specified goals and curated training geometries leave open-world transfer unresolved; false negatives can remove useful action-grasp pairs.
- **Extension / idea:** Learn category-level grasp distributions with calibrated abstention and audit a planner-controlled fraction of rejected branches.

### T089 — Synergistic TAMP with RL Non-Prehensile Actions

- **Novelty:** Inserts a vision-based RL pushing skill when PDDLStream cannot sample a collision-free grasp, while using planner outcomes as reward feedback to avoid irreversible pushes.
- **Limitations:** *Explicit evaluation scope:* one pusher skill and bin-picking experiments in simulation and hardware do not establish general symbolic effects or safety.
- **Extension / idea:** Learn several stochastic contact skills with explicit preconditions/effects, risk bounds, and counterexample-driven interface repair.

### T090 — LLM TAMP for Object Rearrangement

- **Novelty:** Uses an LLM for commonsense rearrangement goals and lets a geometric planner test physical executability.
- **Limitations:** *Inferred:* cultural bias, underspecified preferences, and limited object domains can yield plausible but unwanted arrangements.
- **Extension / idea:** Learn user preferences from correction and ground language claims in visual/execution feedback.

### T091 — Reachability-Tree TAMP

- **Novelty:** Searches abstract choices with Monte Carlo tree search and expands continuous reachability samples hierarchically.
- **Limitations:** *Inferred:* it assumes useful goal and reward definitions, and the empirical benchmarks do not supply broad guarantees.
- **Extension / idea:** Learn goal abstractions and feasibility bounds while retaining systematic exploration.

### T092 — Search-Based TAMP for Agile Hybrid Vehicles

- **Novelty:** Couples discrete driving modes with motion primitives to plan aggressive hybrid maneuvers such as drifting.
- **Limitations:** *Explicit evaluation scope:* a simulated track and engineered primitives depend strongly on the vehicle model.
- **Extension / idea:** Add model uncertainty, real-vehicle validation, and online discovery of useful hybrid modes.

### T093 — Chemistry Lab Automation via Constrained Task and Motion Planning

- **Novelty.** Couples PDDLStream experiment sequencing with spill- and collision-constrained motions, perception, and execution for real chemistry procedures rather than generic pick-and-place.
- **Limitations / extension.** *Inferred from evaluation scope:* the action/tool library and liquid-handling constraints are engineered for a small set of experiments. Extend it with uncertain liquid dynamics, online material-state estimation, and recovery operators that revise both the symbolic experiment and the motion constraints.

### T094 — A Conflict-driven Interface between Symbolic Planning and Nonlinear Constraint Solving

- **Novelty.** Introduces Planning with Nonlinear Transition Constraints and feeds minimal infeasible nonlinear constraint subsets back to symbolic search as reusable conflicts, creating a genuinely bidirectional logic–geometry interface.
- **Limitations / extension.** *Inferred:* conflict quality inherits the local nonlinear solver's failure modes, and enumerating minimal infeasible subsets can itself be costly. A strong extension is certified or probabilistic conflict extraction with learned conflict generalization across related scenes while retaining a sound fallback.

### T095 — Multi-Arm Bin-Picking in Real-Time

- **Novelty.** Integrates a geometry-aware high-level arm/object-selection policy with BIT* refinement, demonstrating real-time cooperative bin picking with up to four arms.
- **Limitations / extension.** *Inferred from the application scope:* the hierarchy is specialized to bin picking and assumes the perception and grasp candidates are usable. Extend it to asynchronous arms, uncertain grasps, and online reassignment when an arm's execution changes another arm's feasible region.

### T096 — Learning to Correct Mistakes

- **Novelty.** Learns a backjumping heuristic that predicts which earlier action caused a downstream geometric dead end, avoiding chronological re-evaluation of every intervening motion choice.
- **Limitations / extension.** *Inferred from the two-task evaluation:* culprit labels and generalization depend on the training distribution, so a wrong jump can waste search. Use calibrated culprit sets, counterfactual data collected online, and a completeness-preserving fallback to ordinary backtracking.

### T097 — Sequence-Based Plan Feasibility Prediction (PIGINet)

- **Novelty.** Uses a Transformer over an initial scene, goal, and whole task sequence to rank action skeletons by continuous refinability, including visual features that support transfer to unseen object categories.
- **Limitations / extension.** *Inferred from the simulated kitchen evaluation:* rankings can be miscalibrated under new geometry, articulation, or contact regimes and do not certify infeasibility. Combine conformal/OOD uncertainty with top-k diverse skeleton selection and retain probabilistically complete refinement behind the learned ordering.

### T098 — Policy-Guided Lazy Search with Feedback

- **Novelty.** Maintains one lazy integrated search whose symbolic choices become progressively informed by motion samples, and combines a learned goal-directed policy with feedback from the current refinement attempt.
- **Limitations / extension.** *Inferred:* evidence is from simulated 7-DoF rearrangement, and policy bias can be harmful out of distribution. Test real dynamic scenes and design safe exploration bonuses or lower-confidence fallback rules that preserve the underlying solver's guarantees.

### T099 — STAP: Sequencing Task-Agnostic Policies

- **Novelty.** Treats learned skill Q-functions as geometric compatibility signals and optimizes continuous skill parameters to maximize estimated success across an entire task-planner-provided sequence.
- **Limitations / extension.** *Inferred:* multiplying per-skill Q-values assumes a useful factorization and relies on calibrated policies; the symbolic sequence is externally supplied. Model correlated failures and belief-state transitions, and jointly search over task order and continuous skill parameters.

### T100 — Task and Motion Informed Trees (TMIT*)

- **Novelty.** Interleaves asymmetric forward and reverse search directly in the hybrid task–motion space, combining makespan-optimal task planning with almost-sure asymptotically optimal motion planning.
- **Limitations / extension.** *Inferred from assumptions and benchmarks:* asymptotic results require robust feasibility and sufficiently regular sampling spaces, while convergence may be slow in contact-rich problems. Develop finite-time suboptimality bounds, kinodynamic transition samplers, and anytime certificates for the current joint plan.

### T101 — Hypergraph-Based Multi-Robot TAMP

- **Novelty.** Represents robots, objects, and robot–object composite spaces as a hypergraph, replacing full composite-state vertices and substantially reducing multi-robot rearrangement search growth.
- **Limitations / extension.** *Inferred:* useful decompositions and transition hyperarcs are still domain-designed, and the evaluation is dominated by rearrangement. Learn or incrementally construct the hypergraph from failed refinements and add explicit concurrency, uncertain duration, and dynamic-obstacle semantics.

### T102 — Multiple Mobile Robot TAMP: A Survey

- **Novelty.** Supplies a multi-axis taxonomy spanning task decomposition/allocation/scheduling, path versus trajectory planning, and fixed versus adaptive execution for multi-mobile-robot TAMP.
- **Limitations / extension.** *Inferred from publication cutoff and scope:* it predates the rapid growth of VLM/LLM guidance, GPU optimization, and newer asynchronous hypergraph methods, and mobile systems dominate. A living survey should attach benchmark instances and update evidence by capability, guarantee, uncertainty type, and real-robot scale.

### T103 — A MIP-Based Approach for Multi-Robot GTAMP

- **Novelty.** Precomputes reachability, occlusion, and handover relations, encodes their precedence structure in a MIP, and uses its solution to guide MCTS over collaborative manipulation plans.
- **Limitations / extension.** *Explicit in the problem definition:* the formulation is synchronous and monotone; *inferred:* precomputed geometric relations can become stale. Extend it to asynchronous non-monotone manipulation with incremental graph updates and uncertainty-aware handover feasibility.

### T104 — Learning Neuro-Symbolic Skills for Bilevel Planning

- **Novelty.** Jointly learns parameterized low-level policies, symbolic operators, and continuous samplers as modular neuro-symbolic skills that a search-then-sample planner can compose on new tasks.
- **Limitations / extension.** *Explicit in the setup:* demonstrations and predicates are supplied; *inferred:* rare effects and execution uncertainty are weakly represented. Jointly invent predicates, learn stochastic effects, and trigger active data collection at symbolic states where refinement uncertainty is highest.

### T105 — Failure Is an Option

- **Novelty.** Incorporates actions that may fail during physical execution into task-and-motion policy computation without assuming perfectly known failure probabilities, using execution experience to improve future choices.
- **Limitations / extension.** *Inferred:* stationary action-level failure estimates can hide context, wear, and correlated failure causes. Learn context-conditioned nonstationary outcome models and plan information-gathering executions subject to explicit risk and recovery budgets.

### T106 — Neural Feasibility Checking

- **Novelty.** Places a visual neural feasibility classifier ahead of expensive motion refinement so obviously infeasible symbolic actions can be deprioritized without repeated IK calls.
- **Limitations / extension.** *Inferred from label construction and simulation:* IK-derived labels omit full-path, dynamics, and contact feasibility, while false negatives can suppress good branches. Use multi-stage labels, calibrated abstention, and a planner-controlled audit rate to preserve completeness.

### T107 — Cooperative TAMP for Multi-Arm Assembly Systems

- **Novelty.** Jointly reasons over assembly precedence, robot assignment, synchronized collision-free motion, and makespan, rather than optimizing the assembly sequence and arm trajectories independently.
- **Limitations / extension.** *Inferred from evaluated structural assembly:* geometry, grasps, and assembly ordering knowledge are largely known offline. Add uncertain action durations, online structural-state sensing, and event-triggered rescheduling with reusable inter-robot conflict cuts.

### T108 — Visually Grounded TAMP for Mobile Manipulation

- **Novelty.** Learns visual grounding scores for object-placement and mobile-base choices, injecting semantic preferences into geometric TAMP so plans are both feasible and context-appropriate.
- **Limitations / extension.** *Inferred:* semantic scores depend on perception and the household training distribution and have no correctness guarantee. Represent score uncertainty, solicit preferences only when they change the selected plan, and close the loop after placement through visual verification.

### T109 — Learning to Ground Objects for Robot TAMP

- **Novelty.** Learns object-level grounding from visual observations to connect symbolic references in a task planner with the physical entities required by motion planning.
- **Limitations / extension.** *Inferred from the closed-domain evaluation:* unseen categories, occlusion, and identity switches can corrupt downstream symbolic state. Maintain distributions over bindings, add active viewpoints as planning actions, and repair the symbolic plan when identity confidence changes.

### T110 — Hierarchical Deliberative-Reactive TAMP in Partially Known Environments

- **Novelty.** Exposes reachability contracts from a reactive vector-field controller to a sampling-based deliberative planner, letting the latter sequence behaviors and goals without constructing every detailed path upfront.
- **Limitations / extension.** *Inferred:* guarantees depend on the reactive planner's environmental and dynamical assumptions, and the demonstrations emphasize navigation/rearrangement. Generalize contracts to manipulation and learned controllers, including probabilistic contract violation and belief-aware fallback behaviors.

### T111 — Learning Geometric Constraints in TAMP

- **Novelty.** Converts semantic and geometric backtracking evidence into transferable constraint primitives, then uses Bayesian optimization to guide continuous-binding search on later tasks in the same workspace.
- **Limitations / extension.** *Inferred:* transfer is strongest when workspaces and constraint structure repeat, and the surrogate may be overconfident after distribution shift. Meta-learn primitives across workspaces and use safe, uncertainty-aware acquisition with planner-generated counterexamples.

### T112 — Guided Imitation of TAMP

- **Novelty.** Builds a high-throughput asynchronous TAMP teacher and a hierarchical policy that imitates its action/control output; partially learned policies in turn accelerate later data generation.
- **Limitations / extension.** *Explicit in reported results:* policies do not reach perfect success (reported averages include 88% and 79%); *inferred:* covariate shift remains. Add intervention-based imitation, planner fallback selected by calibrated policy uncertainty, and recovery demonstrations from off-plan states.

### T113 — Learning to Search in TAMP with Streams

- **Novelty.** Learns search control for a stream-based TAMP solver so prior planning experience prioritizes productive symbolic skeletons and expensive black-box stream evaluations.
- **Limitations / extension.** *Inferred:* learned ordering is tied to the training domains and stream implementations, and poor scores can delay critical samples. Learn cost-aware, uncertainty-calibrated priorities online while periodically forcing coverage of underexplored streams.

### T114 — Safe Legged Navigation in Partially Observable Environments

- **Novelty.** Couples temporal task decisions, terrain/visibility uncertainty, and dynamically feasible legged motion so the robot can reason about safety rather than treating locomotion as a deterministic refinement oracle.
- **Limitations / extension.** *Inferred:* performance depends on terrain-belief and locomotion models and incurs substantial hybrid planning cost. Use learned but verified reachability envelopes, risk calibration from real execution, and receding-horizon belief updates.

### T115 — RHH-LGP

- **Novelty.** Combines receding-horizon Logic-Geometric Programming with task-level heuristics, solving shorter nonlinear programs while retaining long-horizon symbolic guidance.
- **Limitations / extension.** *Inferred:* a short horizon can hide delayed geometric dead ends and each nonlinear solve remains locally sensitive. Adapt the horizon using predicted constraint coupling, cache reusable partial solutions, and invoke global repair when repeated local failures share a conflict.

### T116 — Discovering State and Action Abstractions for Generalized TAMP

- **Novelty.** Learns both abstract state predicates and action structure from continuous experience so one compact TAMP model can generalize compositionally across object counts and goals.
- **Limitations / extension.** *Inferred:* abstraction quality depends on representative, mostly clean training transitions and can silently merge states with different future feasibility. Drive representation refinement with counterexamples and attach uncertainty to predicates and effects.

### T117 — Representation, Learning, and Planning Algorithms for GTAMP

- **Novelty.** Unifies randomized heuristic search, a learned rank function for discrete actions, and learned continuous samplers in an object-centric GTAMP representation designed for data-efficient generalization.
- **Limitations / extension.** *Explicit in scope:* the framework targets geometric rearrangement; *inferred:* quasi-static object models and manually selected predicates limit transfer. Extend the representation to contact modes, dynamics, and belief over object geometry while preserving the planner fallback.

### T118 — Anytime Hierarchical Stochastic TAMP

- **Novelty.** Builds executable stochastic task-and-motion policies incrementally, allocating refinement effort to likely contingencies and improving coverage in an anytime fashion.
- **Limitations / extension.** *Inferred:* policy branching and repeated continuous refinement grow rapidly with horizon and assumed outcome models. Compress equivalent belief/policy branches and learn outcome probabilities online with risk-sensitive refinement scheduling.

### T119 — TAMP with Estimated Affordances for Unknown Objects

- **Novelty.** Estimates affordances for previously unseen objects and exposes them as probabilistic geometric interfaces to a long-horizon task-and-motion planner.
- **Limitations / extension.** *Inferred:* incorrect affordances can invalidate many downstream steps and the skill vocabulary remains fixed. Plan active tests/viewpoints that reduce affordance uncertainty, propagate posterior risk through the whole skeleton, and learn new parameterized skills when no existing refinement fits.

### T120 — Active Learning of Abstract Plan Feasibility

- **Novelty.** Selects informative action sequences for self-supervised execution, exploiting an infeasible-subsequence property to learn an abstract-plan feasibility predictor with fewer robot trials.
- **Limitations / extension.** *Explicit in evaluation scope:* the real study uses a stacking domain and hundreds of interactions. Extend to safe active learning across several skills and use causal failure representations so knowledge transfers beyond sequences that look syntactically similar.

### T121 — Long-Horizon Multi-Robot Construction Assembly

- **Novelty.** Integrates construction order, multi-robot manipulation modes, and composite collision-free trajectories, using a long-horizon solver tailored to tightly coupled structural assembly.
- **Limitations / extension.** *Inferred:* the design, part geometry, and feasible manipulation modes are supplied, and most planning is offline. Add tolerance-aware state estimation, uncertain-duration scheduling, and incremental replanning when the as-built structure deviates from CAD.

### T122 — Neuro-Symbolic Relational Transition Models (NSRTs)

- **Novelty.** Learns relational symbolic operators, neural low-level transitions, and action samplers in one model that supports outer symbolic search and inner continuous refinement across varying object counts.
- **Limitations / extension.** *Explicit in the setup:* predicates are given; *inferred:* learned effects are chiefly deterministic and can miss rare failures. Jointly invent predicates, model stochastic/multimodal outcomes, and use execution counterexamples to split over-coarse operators.

### T123 — Combining TAMP: Challenges and Guidelines

- **Novelty.** Organizes TAMP design around five practical questions: abstraction, joint representation/reasoning, learning, online planning, and uncertain perception, connecting algorithmic choices to industrial requirements.
- **Limitations / extension.** *Inferred:* it is a selective guidelines review rather than a reproducible systematic search and predates recent foundation-model methods. Turn the questions into an executable decision matrix tied to benchmark metadata and update it as a living evidence review.

### T124 — Counterexample-Guided Repair for Symbolic-Geometric Action Abstractions

- **Novelty.** Uses geometric refinement counterexamples to repair an initially incorrect symbolic abstraction, importing CEGAR-style feedback into task-and-motion domain construction.
- **Limitations / extension.** *Inferred:* repairs may overfit individual failures, and termination or minimality can depend on the abstraction language. Learn generalized but proof-checked repairs, select the next counterexample actively, and preserve previously established feasible behavior.

### T125 — Fast MILP-Based TAMP for Pick-and-Place

- **Novelty.** Encodes action choice and collision-free routing in one MILP with both hard and soft constraints, enabling optimization over feasibility and preference rather than sequentially solving them.
- **Limitations / extension.** *Inferred:* linearized/discretized collision models trade geometric fidelity for speed and scale poorly with resolution and object count. Add lazy geometric cuts, continuous trajectory polishing, and warm-started replanning after execution updates.

### T126 — Human-Motion Prediction and LGP

- **Novelty.** Embeds hierarchical human-motion predictions inside Logic-Geometric Programming so robot task order and trajectories minimize interference over a multi-step human–robot task.
- **Limitations / extension.** *Inferred:* forecasts can be wrong when humans react to the robot, and uncertainty is not fully propagated through task choice. Plan over interactive multimodal human predictions and expose legibility or clarification as first-class actions.

### T127 — Reactive TAMP under Temporal Logic Specifications

- **Novelty.** Connects temporal-logic task requirements to online task-and-motion execution, monitoring environmental changes and replanning while preserving the specification-level structure.
- **Limitations / extension.** *Inferred:* proposition grounding and action models are assumed reliable, and automaton products can grow rapidly. Add uncertain semantic monitoring, compositional automata, and a safety shield for the interval before a revised plan is available.

### T128 — ThreeDWorld Transport Challenge

- **Novelty.** Defines a physically realistic, visually grounded long-horizon transport benchmark that forces systems to combine object choice, navigation/manipulation order, and continuous interaction rather than solve isolated skills.
- **Limitations / extension.** *Inferred:* results can specialize to the simulator, action API, and challenge task distribution. Add hidden dynamics splits, standardized symbolic/geometric difficulty measures, real-robot correspondences, and explicit recovery scoring.

### T129 — Extended Tree Search for Robot TAMP

- **Novelty.** Extends tree search to interleave discrete action selection and continuous parameter binding, using rollout information to focus refinement on promising hybrid branches.
- **Limitations / extension.** *Inferred:* black-box rollouts and continuous branching can be sample hungry, with limited guarantees in narrow feasible sets. Combine learned proposal distributions with coverage-enforcing exploration and reusable infeasibility explanations.

### T130 — Learning Symbolic Operators for TAMP

- **Novelty.** Induces STRIPS-like preconditions and effects from continuous transition data and combines them with learned samplers, reducing manual construction of the symbolic half of a bilevel planner.
- **Limitations / extension.** *Explicit in the setup:* predicates are provided; *inferred:* incomplete demonstrations can omit rare preconditions or outcomes. Couple operator induction with active counterexample generation, predicate invention, and stochastic effects.

### T131 — Learning Efficient Constraint Graph Sampling

- **Novelty.** Uses MCTS to learn an efficient variable-subset assignment order for factored nonlinear constraint graphs, producing diverse feasible mode-switch configurations for sequential manipulation.
- **Limitations / extension.** *Inferred:* the learned order is problem-distribution dependent and focuses on waypoint feasibility rather than complete TAMP search. Adapt ordering online as constraints change and jointly allocate effort between mode samples, skeletons, and path refinement.

### T132 — SyDeBO

- **Novelty.** Embeds symbolic decisions directly in a bilevel optimization for long-horizon dynamic manipulation, allowing discrete mode selection and continuous trajectory decisions to influence each other.
- **Limitations / extension.** *Inferred:* nonconvex lower-level optimization is initialization sensitive, symbolic modes are engineered, and global guarantees are limited. Learn mode proposals but validate them with relaxations/cuts, and extend to contact uncertainty and receding-horizon execution.

### T133 — Integrated Task and Motion Planning (Annual Review)

- **Novelty.** Provides the field's canonical representation-and-algorithm taxonomy, explaining how sampling, optimization, and search approaches couple symbolic actions with continuous constraints.
- **Limitations / extension.** *Inferred from the 2020 cutoff:* it cannot cover the later waves of foundation-model guidance, learned abstractions, GPU TAMP, or modern temporal multi-robot solvers. A versioned living review should preserve this taxonomy while attaching reproducible benchmarks and guarantee metadata.

### T134 — Planning with Learned Object Importance

- **Novelty.** Learns a graph-neural object-importance model that selects a small relevant subproblem before TAMP search, enabling generalization from small training scenes to larger object sets.
- **Limitations / extension.** *Inferred:* omitting a seemingly irrelevant object can remove all solutions, particularly under unfamiliar relational structure. Use uncertainty-aware supersets, iterative reintroduction after failed refinement, and certificates explaining why excluded objects cannot matter.

### T135 — Receding-Horizon TAMP in Changing Environments

- **Novelty.** Repeatedly couples task and motion replanning over a finite horizon so a robot can respond to environmental changes without rebuilding a full long-horizon plan every cycle.
- **Limitations / extension.** *Inferred:* horizon myopia can choose locally cheap states that block future actions, and changes must be detected correctly. Learn a terminal value over geometric arrangements and plan in belief space with event-triggered horizon expansion.

### T136 — Deep Visual Reasoning

- **Novelty.** Predicts promising discrete action sequences directly from an initial scene image and goal, using them to bypass much of the combinatorial LGP skeleton search while retaining continuous optimization for execution.
- **Limitations / extension.** *Inferred:* a sequence prediction is neither a feasibility certificate nor robust to substantial visual/domain shift. Generate calibrated diverse top-k sequences, verify every candidate geometrically, and train from the verifier's counterexamples.

### T137 — Learning Compositional Models of Robot Skills

- **Novelty.** Learns reusable skill precondition/effect models and continuous parameter samplers that can be composed by a TAMP solver into action sequences not seen during training.
- **Limitations / extension.** *Inferred:* learned skill models assume sufficient coverage and can compound small model errors over long horizons. Maintain epistemic uncertainty, request targeted trials at plan-critical boundaries, and learn explicit recovery transitions between skills.

### T138 — Deep Visual Heuristics

- **Novelty.** Predicts the feasibility of mixed-integer manipulation-planning choices from images, using the classifier as a heuristic to avoid solving many doomed nonlinear programs.
- **Limitations / extension.** *Inferred:* the visual predictor is evaluated on a constrained manipulation family and can make unsafe false-negative/false-positive judgments. Add calibrated abstention, geometry-aware representations, and a solver fallback that audits low-confidence pruning.

### T139 — Arranging Test Tubes in Racks with CTAMP

- **Novelty.** Integrates discrete tube relocation/order decisions with grasp and collision-free arm motions, handling the temporary placements needed to rearrange densely occupied racks.
- **Limitations / extension.** *Inferred:* rack geometry, tube poses, and manipulation actions are tightly specialized and largely known. Add perception uncertainty, laboratory safety constraints, and closed-loop recovery from failed grasps or displaced tubes.

### T140 — Integrating CTAMP with Compliant Control

- **Novelty.** Connects an open-loop combined task-and-motion plan to compliant controllers for contact-sensitive execution, narrowing the common gap between geometric feasibility and physical insertion/contact success.
- **Limitations / extension.** *Inferred:* controller success is summarized through hand-designed action interfaces and demonstrated on a narrow manipulation setup. Learn symbolic outcome models from force/torque traces and replan task choices when compliance reveals an unmodeled contact state.

### T141 — Robust TAMP for Architectural Construction

- **Novelty.** Applies LGP-style joint sequencing and geometric optimization to long-horizon architectural assembly, including structural/task constraints that make the construction order inseparable from robot motion.
- **Limitations / extension.** *Inferred:* it assumes accurate CAD, material geometry, and action models, while construction tolerances accumulate. Fuse as-built sensing into the symbolic/geometric state and plan tolerance-aware corrective or rework actions.

### T142 — Probabilistic Constrained Manipulation and TAMP

- **Novelty.** Represents uncertain constraints probabilistically inside sequential manipulation optimization, allowing TAMP to trade feasibility likelihood against trajectory cost rather than treat every model quantity as exact.
- **Limitations / extension.** *Inferred:* tractability relies on chosen distributional/constraint approximations and may miss multimodal contact outcomes. Use non-Gaussian belief representations, risk allocation across the action sequence, and posterior updates after each execution phase.

### T143 — Object-Centric TAMP in Dynamic Environments

- **Novelty.** Uses an object-centric representation and local replanning so changes to individual objects update only the affected task and motion constraints instead of invalidating an entire monolithic plan.
- **Limitations / extension.** *Inferred:* object identities, action models, and observed state are assumed reliable, and demonstrations are modest in scale. Extend to open-world object discovery, uncertain tracking, and dependency-aware repair across concurrent actions.

### T144 — Relational Value Functions for Guiding TAMP

- **Novelty.** Encodes occlusion relations as a graph and learns a GNN action-value function that transfers from small GTAMP instances to larger scenes and different geometry.
- **Limitations / extension.** *Inferred:* relational predicates are engineered and evaluation centers on rearrangement; unseen relations can break transfer. Learn predicates jointly from geometry, quantify OOD uncertainty, and retain search coverage when value estimates are unreliable.

### T145 — Dual-Arm TAMP to Use a Suction Cup Tool

- **Novelty.** Plans tool acquisition, transfer/use, and coordinated dual-arm motion as one symbolic-geometric problem, showing that tool-use decisions must be grounded in both arms' reachability.
- **Limitations / extension.** *Inferred:* the tool, workstation, and action schemas are fixed and the execution model is mostly deterministic. Generalize to tool selection and substitution, uncertain attachment, and closed-loop bimanual synchronization.

### T146 — Task-Assisted Motion Planning in Partially Observable Domains

- **Novelty.** Lets high-level task actions alter observability and guide low-level motion, integrating information-relevant decisions with geometric planning in partially observable scenes.
- **Limitations / extension.** *Inferred:* the belief/action model and observation choices are structured by hand and can scale poorly with many hidden variables. Use factored beliefs, learned observation models, and value-of-information bounds to decide when sensing is worth delaying task progress.

### T147 — Anytime Multi-Arm TAMP via Handoffs

- **Novelty.** Extends dRRT*-style composite-space search across manipulation modes and incrementally samples pick/handoff configurations, yielding fast initial multi-arm plans that improve asymptotically.
- **Limitations / extension.** *Inferred:* candidate grasps/handoffs and quasi-static object models are assumed; synchronized composite planning grows with arm count. Add asynchronous temporal coordination, learned transfer proposals, and dynamic execution repair.

### T148 — Anytime Integrated Task and Motion Policies for Stochastic Environments

- **Novelty.** Produces branching task-and-motion policies for stochastic outcomes and refines unresolved contingencies over time, with a probabilistic-completeness argument rather than returning only a nominal plan.
- **Limitations / extension.** *Inferred:* branching explodes with horizon and the outcome model is assumed known enough to prioritize contingencies. Learn probabilities online, merge geometrically equivalent policy branches, and optimize risk or regret rather than only coverage.

### T149 — Asymptotic Optimality in Integrated TAMP

- **Novelty.** Sharpens asymptotic-optimality theory for multimodal TAMP, showing standard connection radii can suffice and extending clearance arguments to isolated nonsmooth transition states such as grasps.
- **Limitations / extension.** *Inferred:* the proof needs robust regularity conditions and says little about finite-time convergence in difficult manipulation geometry. Derive finite-sample rates, account for optimization-based transition generators, and test contact-rich mode boundaries.

### T150 — Learning Feasibility in Tabletop Environments

- **Novelty.** Learns a motion-feasibility classifier from minimal exemplar scenes and uses conservative geometric approximations to order constraint-based TAMP search without making classifier correctness a hard requirement.
- **Limitations / extension.** *Explicit in scope:* the setting is a fixed robot in tabletop scenes; *inferred:* primitive shape approximations omit richer geometry and dynamics. Learn relational 3D feasibility with calibrated uncertainty while preserving the robust ordering/fallback principle.

### T151 — TMP-RL for Robust Mobile-Robot Decisions

- **Novelty.** Nests a task–motion planning loop inside an execution-learning loop, using real experience to update abstract action values while TAMP supplies a safe, data-efficient initial policy.
- **Limitations / extension.** *Inferred from the office-navigation evaluation:* model-free learning remains sample intensive and cost updates may not transfer to new failure causes. Use model-based uncertainty, safety-constrained exploration, and context-conditioned values that distinguish geometry from transient execution conditions.

### T152 — Reactive Whole-Body Dynamic Locomotion TAMP

- **Novelty.** Combines a temporal-logic game, a library of robust locomotion transition models, and whole-body trajectory generation, including replanning after environmental events or large perturbations.
- **Limitations / extension.** *Inferred:* the motion-template library and abstraction are hand-built and the original evidence is simulation-heavy. Automatically certify new learned locomotion modes and validate the full reactive stack on hardware under perception delay.

### T153 — Score-Space Guidance for TAMP

- **Novelty.** Represents a new planning instance by the scores of solutions attempted so far, enabling transfer of search constraints from previously solved instances without requiring a fixed-size raw scene encoding.
- **Limitations / extension.** *Inferred:* informative score-space coordinates require attempted solutions, and similarity may be misleading under a new constraint regime. Combine relational scene embeddings with online score evidence and use uncertainty to relax transferred constraints.

### T154 — Admissible Abstractions for Near-Optimal TAMP

- **Novelty.** Formalizes admissible angelic abstractions and derives lower/upper cost bounds that accelerate hybrid planning while retaining near-optimality guarantees.
- **Limitations / extension.** *Inferred:* abstraction and metric/topological bounds are domain-derived and demonstrated in limited continuous planning domains. Automate abstraction synthesis from geometric decompositions and extend admissibility to uncertainty, dynamics, and learned bounds.

### T155 — Scalable TAMP from STL Specifications

- **Novelty.** Separates discrete task choices and continuous trajectory synthesis on the fly using SMT and LP, with soundness/completeness for nonconvex STL specifications under the modeled system class.
- **Limitations / extension.** *Inferred from formulation:* guarantees rely on discretized, linear/polyhedral dynamics and predicates; nonlinear contact creates a different problem. Add abstraction-refinement for nonlinear dynamics and exploit compositional STL structure to control solver growth.

### T156 — Active Model Learning and Diverse Action Sampling

- **Novelty.** Actively learns sensorimotor primitive applicability with Gaussian processes and introduces diverse continuous sampling so newly learned skills can be composed efficiently by TAMP.
- **Limitations / extension.** *Inferred:* robot experiments are costly and GP assumptions become restrictive in high-dimensional or discontinuous contact domains. Use safe batch active learning, structured kernels/representations, and planner-derived value of information.

### T157 — PDDLStream

- **Novelty.** Extends PDDL with black-box streams that generate and certify continuous objects, and introduces optimistic adaptive planning that balances new skeleton exploration against binding refinement.
- **Limitations / extension.** *Inferred:* users must supply correct symbolic schemas and effective samplers; black-box costs and failures can dominate. Learn streams and their costs from experience, but validate certifications and retain systematic sampling for probabilistic completeness.

### T158 — Anytime Task and Motion MDPs

- **Novelty.** Computes consistent task-and-motion policies for the most likely stochastic outcomes first, then incrementally expands the policy with a probabilistic-completeness guarantee.
- **Limitations / extension.** *Inferred:* the MDP and outcome probabilities are known, while continuous refinement over many branches remains expensive. Extend to POMDP beliefs, learned transition uncertainty, and branch-and-bound allocation of refinement effort by risk contribution.

### T159 — Sampling-Based Factored TAMP

- **Novelty.** Formalizes hybrid planning as a factored transition system, characterizes robust feasibility on lower-dimensional constraint manifolds, and gives domain-independent probabilistically complete algorithms based on conditional samplers.
- **Limitations / extension.** *Explicit in the interface:* suitable conditional samplers are inputs; *inferred:* poor samplers make guarantees impractical. Synthesize sampler structure from constraint graphs, learn proposals without losing coverage, and extend the theory to stochastic execution.

### T160 — Conditional TAMP through an Effort-Based Approach

- **Novelty.** Triggers replanning not only when a path becomes impossible but whenever a conditional alternative can reduce execution effort, making cost adaptation part of online TAMP.
- **Limitations / extension.** *Explicit:* the paper describes preliminary work and says experiments were in progress; completeness/scalability were expectations rather than established empirical results. Formalize those claims, evaluate nonstationary effort estimates, and compare event-triggered versus periodic replanning.

### T161 — Manipulator TAMP with Metric Temporal Logic

- **Novelty.** Uses MTL to express timed manipulation tasks, a high-level MILP to choose a candidate task/pose realization, and gradient-based optimization for the arm trajectory.
- **Limitations / extension.** *Inferred from the Baxter simulation and hierarchy:* a fixed horizon, supplied grasp/place primitives, and local trajectory optimization limit robustness. Add receding-horizon monitoring, contact-aware refinement, and conflict feedback from the motion optimizer to the MILP.

### T162 — TAMP as Classical AI Planning

- **Novelty.** Compiles sampled configurations, collision/motion relations, functions, and state constraints into a finite classical planning problem; the compilation is sound and probabilistically complete as samples grow.
- **Limitations / extension.** *Inferred:* compilation can become very large and calls motion planners before discrete search knows which relations matter. Build the compilation lazily, learn which samples/relations are relevant, and preserve the soundness/completeness argument under incremental growth.

### T163 — Multi-Bound Tree Search for LGP

- **Novelty.** Combines branch-and-bound and MCTS with several increasingly expensive LGP relaxations/bounds, directing search through cooperative manipulation skeletons before solving full paths.
- **Limitations / extension.** *Explicit in characterization:* it is an approximate solver; *inferred:* nonlinear local minima and one Baxter–human scenario limit generality. Develop admissible or uncertainty-aware bounds, parallelize relaxation evaluation, and learn when each bound is worth computing.

### T164 — Neural Networks and Tree Search for TAMP

- **Novelty.** Integrates learned low-level control options and high-level option priors with MCTS under LTL task constraints, demonstrating hybrid task/motion decision making in interactive autonomous driving.
- **Limitations / extension.** *Inferred:* learned options do not inherit formal LTL guarantees, and evidence is simulation-specific. Wrap options in reachability/safety certificates, quantify model uncertainty in tree search, and evaluate rare adversarial traffic interactions.

### T165 — STRIPS Planning in Infinite Domains (STRIPStream)

- **Novelty.** Extends STRIPS with black-box streams that enumerate continuous objects and certify static predicates, reducing infinite-domain TAMP to a sequence of finite planning problems.
- **Limitations / extension.** *Inferred:* performance is sensitive to stream ordering and repeated finite replanning, and modeling still requires hand-written generators. The natural extension is optimistic/lazy adaptive stream evaluation, cost models, and learned proposals backed by exhaustive stream coverage.

## Classic-era core analysis

### T166 — FFRob: Leveraging Symbolic Planning for Efficient Task and Motion Planning

- **Novelty:** Defines Extended Action Specification and transfers FF/delete-relaxation heuristics into TAMP, using batch-sampled manipulation primitives and conditional multi-query roadmaps; also proves probabilistic completeness and finite expected runtime under its assumptions.
- **Limitation / extension:** **Inferred:** the guarantees depend on robust feasibility and nonzero-probability samplers, while the experiments assume known, mostly rigid and static worlds. **Extension idea:** learn calibrated conditional samplers and roadmap reuse policies while preserving a nonzero uniform-sampling component for completeness under perception uncertainty.

### T167 — Incremental Task and Motion Planning: A Constraint-Based Approach

- **Novelty:** IDTMP couples increasing task horizons to increasing motion-planning effort and incrementally pushes motion infeasibility constraints into an SMT task planner, obtaining probabilistic completeness and strong empirical scaling.
- **Limitation / extension:** **Inferred:** difficult geometric failures can still cause repeated expensive checks, and completeness assumes robust motion-level solutions and accurate scene models. **Extension idea:** learn reusable, minimally scoped conflict clauses and schedule validation by expected information gain, with dynamic invalidation during execution.

### T168 — Integrated TAMP for Multiple Robots under Path and Communication Uncertainties

- **Novelty:** Introduces a task-reachability graph updated by sampling-based path costs and an HMM belief, then uses an MDP for task ordering plus a deadlock-free coordination procedure.
- **Limitation / extension:** **Inferred:** the task layer is primarily location visitation and uncertainty is concentrated in path costs/communications, not manipulation outcomes or open-world goals. **Extension idea:** generalize the belief state to action success, resource contention, and intermittent observations, with decentralized belief fusion and receding-horizon reassignment.

### T169 — Formal Design of Robot Integrated Task and Motion Planning

- **Novelty:** CoSMoP composes motion primitives modeled as hybrid automata, proves local safety with differential dynamic logic, and uses SMT with geometric constraints to build globally safe task-motion plans around moving obstacles.
- **Limitation / extension:** **Inferred:** verified primitives and their contracts are manually supplied, the task horizon is bounded, and validation is illustrative rather than large-scale. **Extension idea:** automatically synthesize or repair primitive contracts from data, then apply counterexample-guided abstraction refinement to scale correct-by-construction TAMP.

### T170 — Combining TAMP: A Culprit Detection Problem

- **Novelty:** Recasts geometric failure explanation as finding a small culprit set of task-level constraints, allowing the symbolic planner to reject families of infeasible plans rather than only one skeleton.
- **Limitation / extension:** **Inferred:** usefulness depends on how faithfully the geometric constraints expose causes; deterministic, hand-modeled constraints may yield weak or expensive explanations. **Extension idea:** cache generalized probabilistic culprits with confidence, retract them when perception changes, and learn which explanation granularity maximizes downstream pruning.

### T171 — Task and Motion Policy Synthesis as Liveness Games

- **Novelty:** Moves from one-shot plans to reactive task-motion policies by casting repeated progress and environment interaction as a liveness game over a discrete abstraction connected to continuous motion.
- **Limitation / extension:** **Inferred:** finite abstractions and game graphs can grow rapidly, and correctness applies only if the abstraction and motion contracts are sound. **Extension idea:** use CEGAR to refine only motion-relevant regions and synthesize risk-aware strategies over learned probabilistic transition contracts.

### T172 — Guided Search for TAMP Using Learned Heuristics

- **Novelty:** Learns a heuristic to rank task-and-motion search choices from prior solved problems, demonstrating that experience can guide the coupled search without replacing geometric validation.
- **Limitation / extension:** The paper's hand-designed feature representation limits transfer to new object counts, geometries, and domains. **Extension idea:** use permutation-equivariant scene/action encoders, calibrated uncertainty, and a safe fallback heuristic so guidance can generalize without suppressing novel feasible branches.

### T173 — Task Planning Using Physics-Based Heuristics on Manipulation Actions

- **Novelty:** Injects physics-based feasibility and effort estimates for manipulation actions into task-level heuristic search instead of treating actions as purely symbolic.
- **Limitation / extension:** **Inferred:** repeated simulation is costly and sensitive to friction, mass, and contact-model error; demonstrated action families are narrow. **Extension idea:** combine online system identification with a conservative learned physics surrogate and selectively invoke high-fidelity simulation near decision boundaries.

### T174 — Humanoid Manipulation Planning Using Backward-Forward Search

- **Novelty:** Adapts backward-forward manipulation search to humanoid whole-body reachability, exploiting goal-side constraints to reduce expensive geometric branching.
- **Limitation / extension:** The speed-oriented search gives up probabilistic-completeness guarantees and may return low-quality seed motions. **Extension idea:** hand the discrete/geometric skeleton to whole-body trajectory optimization and retain an anytime completeness-preserving background search.

### T175 — Sequential Quadratic Programming for Task Plan Optimization

- **Novelty:** Optimizes continuous parameters across an entire task-plan skeleton with SQP, allowing later geometric costs and constraints to influence earlier choices rather than refining each action independently.
- **Limitation / extension:** **Inferred:** SQP is local, depends on differentiable models and a fixed discrete skeleton, and can struggle with contact discontinuities. **Extension idea:** couple multi-start SQP to skeleton search, use smooth/contact-complementarity relaxations, and learn warm starts with feasibility certificates.

### T176 — Asymptotically Optimal Planning under Piecewise-Analytic Constraints

- **Novelty:** Gives an asymptotically optimal sampling framework for hybrid manipulation domains whose active differential constraints change with grasp/release modes, without requiring a hand-coded symbolic abstraction.
- **Limitation / extension:** **Inferred:** the result assumes a locally complete transition planner and positive-probability access to reachable subsets; convergence can be impractically slow in high dimensions. **Extension idea:** learn mode-transition proposals and admissible lower bounds while retaining unbiased samples, then extend analysis to stochastic contact and belief-space costs.

### T177 — Geometric Backtracking for Combined TAMP in Robotic Systems

- **Novelty:** Develops symbolic-geometric backtracking in which geometric constraint propagation identifies which earlier task choices caused failure, rather than restarting from the most recent action.
- **Limitation / extension:** **Inferred:** interval/constraint abstractions can be conservative or weak in tightly coupled high-dimensional geometry and assume a mostly known deterministic world. **Extension idea:** derive nonlinear unsatisfiable cores from modern trajectory optimizers and maintain retractable nogoods under changing perception.

### T178 — Logic-Geometric Programming

- **Novelty:** Formulates a task sequence as a switch in the constraints of one trajectory-level nonlinear program and uses multiple bounds/approximations to let continuous optimization guide logical search.
- **Limitation / extension:** **Inferred:** nonconvex optimization can reject a feasible skeleton through a poor local minimum, while logic, features, and contact modes remain engineered. **Extension idea:** combine certified convex relaxations, diverse learned initializations, and belief-aware constraints, with explicit distinction between local failure and proven infeasibility.

### T179 — Backward-Forward Search for Manipulation Planning

- **Novelty:** Searches simultaneously from initial and goal conditions so strong goal-side geometric constraints prune manipulation choices earlier than pure forward refinement.
- **Limitation / extension:** **Inferred:** effectiveness depends on reversible/bridgeable manipulation primitives and structured goal constraints, with evaluation centered on rigid quasi-static manipulation. **Extension idea:** learn bidirectional bridge samplers for contact-rich and articulated tasks, together with a certificate when a backward state is dynamically reachable.

### T180 — Modular TAMP in Belief Space

- **Novelty:** Separates task, motion, and uncertainty reasoning into modular components while planning over beliefs, letting task choices depend on uncertain geometric state rather than only a maximum-likelihood world.
- **Limitation / extension:** **Inferred:** approximate belief representations and supplied sensing/action models can miss multimodality and rare failures, and belief-space refinement is expensive. **Extension idea:** use particles or mixture beliefs only on task-relevant variables, add information-gathering actions, and calibrate model error online.

### T181 — Extending the Knowledge of Volumes Approach

- **Novelty:** Makes KVP more computationally useful by deriving efficient geometric predicates from knowledge of occupied, swept, and interaction volumes and exposing them to a symbolic task planner.
- **Limitation / extension:** **Inferred:** volumes and predicates are hand-authored approximations that can be conservative and brittle to perception or articulated geometry. **Extension idea:** infer probabilistic volumes from sensor data, refine only predicates that affect the incumbent plan, and attach explanation/certainty metadata.

### T182 — Hierarchical Planning for Multi-Contact Non-Prehensile Manipulation

- **Novelty:** Extends hierarchical manipulation planning beyond pick-and-place to multi-contact, non-prehensile action sequences, jointly reasoning about discrete contact modes and continuous paths.
- **Limitation / extension:** **Inferred:** contact primitives and physics assumptions are domain-specific, and model error can invalidate long open-loop sequences. **Extension idea:** discover contact modes from demonstrations, optimize robust funnels for each mode, and replan from force/vision feedback.

### T183 — Symbolic-Geometric Planning for Human-Aware Plans

- **Novelty:** Incorporates human-oriented geometric criteria into the combined search so symbolic alternatives can be compared by reachability, visibility, effort, and human-aware placement consequences.
- **Limitation / extension:** **Inferred:** the human model and cost tradeoffs are manually specified and tested in limited interaction scenarios. **Extension idea:** learn personalized latent preferences with uncertainty, include human response predictions, and expose Pareto alternatives instead of a single scalarized plan.

### T184 — Hybrid Diagnostic Reasoning for Cognitive Factories

- **Novelty:** Integrates ASP-based diagnosis with task-motion plan monitoring so multi-robot factory execution can identify failures and select recovery actions rather than merely re-run a nominal planner.
- **Limitation / extension:** **Inferred:** diagnosis is bounded by engineered fault hypotheses and assumes the monitoring stack observes enough evidence to distinguish them. **Extension idea:** combine symbolic explanations with probabilistic fault beliefs, actively choose diagnostic motions, and feed learned recurring failures back into planning.

### T185 — Efficiently Combining TAMP Using Geometric Constraints

- **Novelty:** Propagates interval-valued geometric constraints into task search to detect impossible combinations earlier and reduce blind task-plan/geometric-plan alternation.
- **Limitation / extension:** **Inferred:** interval bounds weaken with dimensionality and correlation, while accurate action-specific constraints require engineering. **Extension idea:** mix cheap intervals with selectively invoked nonlinear relaxations and learn reusable generalized conflict constraints across scenes.

### T186 — Planner-Independent Interface Layer

- **Novelty:** Defines an extensible interface through which a classical planner can introduce continuous symbols and query external geometric procedures, decoupling the integration method from any one task or motion planner.
- **Limitation / extension:** **Inferred:** black-box procedures may be expensive, partial, or nondeterministic, and the interface itself does not guarantee completeness without disciplined generators. **Extension idea:** standardize typed multi-result streams with provenance, confidence, caching, and fairness contracts for parallel refinement.

### T187 — FFRob: An Efficient Heuristic

- **Novelty:** Conditionalizes a multi-query roadmap on movable-object placements and embeds its reachability structure in an FF-style heuristic, tightly coupling symbolic relaxed-plan estimates to geometry.
- **Limitation / extension:** **Inferred:** the heuristic operates over a finite sampled discretization and assumes known static geometry; its quality can collapse when useful grasps/placements are not sampled. **Extension idea:** adapt sample allocation to relaxed-plan bottlenecks and blend learned proposals with completeness-preserving exploration.

### T188 — Constraint-Based Sequential Manipulation Planning

- **Novelty:** Searches symbolic plan skeletons while postponing poses, grasps, configurations, and paths into one geometric CSP, preserving cross-action constraints instead of refining steps independently.
- **Limitation / extension:** **Inferred:** discrete candidate sets and expensive path-existence predicates determine scalability, and a failed CSP may not explain which skeleton choice to revise. **Extension idea:** incrementalize the CSP, generate conflict cores, and add continuous optimization only for constraints that survive coarse filtering.

### T189 — SMT-Based Synthesis from Plan Outlines

- **Novelty:** Robosynth combines a user-provided plan outline, logical requirements, a placement-graph abstraction, and SMT to synthesize an executable integrated plan from a constrained family.
- **Limitation / extension:** **Inferred:** completeness is relative to the outline and finite placement graph, so a bad outline or missing graph edge excludes valid behaviors. **Extension idea:** infer and relax outlines automatically, use CEGAR to enrich placement graphs, and verify generated plans during execution.

### T190 — Symbolic-Geometric Backtracking for HRI

- **Novelty:** Interleaves symbolic/HTN refinement with a geometric task planner and propagates geometric failures backward to the relevant symbolic choice in human-robot tasks.
- **Limitation / extension:** **Inferred:** decomposition methods, geometric candidates, and human-oriented constraints are engineered, and failure explanations may be solver-specific. **Extension idea:** learn backjump targets and HTN methods from traces while representing human-state uncertainty explicitly.

### T191 — Hybrid Reasoning for Heterogeneous Robot Teams

- **Novelty:** Couples ASP-based global task allocation/sequencing with geometric feasibility checks to find feasible, cost-aware plans for multiple heterogeneous robot teams.
- **Limitation / extension:** **Inferred:** centralized reasoning, known capabilities, and synchronous/static assumptions limit scaling and resilience to execution drift. **Extension idea:** decentralize constraint exchange, support asynchronous durative actions, and reallocate tasks from live capability beliefs.

### T192 — Hybrid Reasoning for Geometric Rearrangement

- **Novelty:** Uses a high-level ASP model and geometric reasoning together to decide object order and placements in cluttered rearrangement rather than accepting a geometrically blind symbolic sequence.
- **Limitation / extension:** **Inferred:** evaluations use bounded tabletop-style settings and discretized placements, with limited treatment of uncertainty or nonmonotone recovery. **Extension idea:** generate placements continuously, learn obstruction graphs, and maintain contingency branches for perception and grasp failures.

### T193 — Extensible Software Architecture for Composing Motion and Task Planners

- **Novelty:** Identifies reusable software abstractions for connecting task planners, sampling-based motion planners, controllers, collision checking, and scene/state propagation in one platform.
- **Limitation / extension:** **Inferred:** the main contribution is architecture rather than a new planning guarantee, and performance evidence is tied to a small set of integrations. **Extension idea:** define reproducible interface conformance tests, planner-agnostic benchmark traces, and modern distributed/accelerator-aware execution APIs.

### T194 — Integrated Task and Motion Planning in Belief Space

- **Novelty:** Extends hierarchical planning in the now to uncertain manipulation by integrating perception, state estimation, information gathering, task choice, and geometric motion in a common belief-aware hierarchy.
- **Limitation / extension:** **Inferred:** compact/Gaussian belief assumptions and supplied observation/action models can miss severe multimodality, while long-horizon belief planning grows quickly. **Extension idea:** factor particles over task-relevant variables, learn observation models, and invoke active perception only where it can change the discrete plan.

### T195 — Hierarchical Manipulation with Diverse Actions

- **Novelty:** Frames diverse prehensile and non-prehensile manipulation as a multimodal problem and uses a hierarchy to avoid searching all robot/object degrees of freedom at once.
- **Limitation / extension:** **Inferred:** useful action types and propagation routines are supplied by hand, and the physical model is largely deterministic. **Extension idea:** discover mode abstractions and transition proposals from demonstrations, then attach robust feedback policies and uncertainty-aware costs.

### T196 — KVP: A Knowledge of Volumes Approach

- **Novelty:** Represents spatial action requirements through declarative volumes, enabling a task planner to reason about reachability, swept occupancy, visibility, and placement without embedding full geometry in the symbolic state.
- **Limitation / extension:** **Inferred:** precomputed volume models can over- or under-approximate actual articulated motion and require substantial domain engineering. **Extension idea:** estimate volumes online from scene geometry and robot reachability, refine them lazily, and propagate uncertainty into symbolic predicates.

### T197 — Lazy Evaluation and Subsumption Caching

- **Novelty:** Delays costly semantic-attachment calls until search needs them and reuses results not only for identical queries but for logically subsumed geometric queries.
- **Limitation / extension:** **Inferred:** cache utility depends on the chosen query abstraction, and stale results become unsafe after scene changes. **Extension idea:** learn the value of each evaluation, maintain dependency-aware invalidation, and share certified caches across related problem instances.

### T198 — Classical Planners for Continuous Operators

- **Novelty:** Uses first-order representations and synchronization with continuous procedures so a classical planner can introduce previously unknown continuous values and revise symbolic structure from geometric results.
- **Limitation / extension:** **Inferred:** continuous generators and their symbolic interfaces remain hand-engineered; fairness and completeness depend on how values are produced. **Extension idea:** formalize typed generator contracts, learn proposals from experience, and reserve unbiased sampling to retain broad coverage.

### T199 — Optimization in the Now

- **Novelty:** Performs dynamic peephole optimization over the currently relevant portion of a hierarchical plan during execution, reducing commitment to distant geometric details.
- **Limitation / extension:** **Inferred:** short optimization windows can miss global interactions and rely on stable execution/model assumptions beyond the window. **Extension idea:** adapt horizon length to predicted coupling and uncertainty, and retain multiple future skeletons when the commitment risk is high.

### T200 — Foresight and Reconsideration in Hierarchical Planning and Execution

- **Novelty:** Adds bounded lookahead and explicit reconsideration to hierarchical execution so the robot can avoid some premature geometric commitments and revise plans as the world evolves.
- **Limitation / extension:** **Inferred:** manually chosen horizons and triggers can either waste planning effort or react too late. **Extension idea:** learn value-of-replanning and failure-hazard models, with safety-triggered replanning as a nonlearned fallback.

### T201 — Interface for Interleaved Symbolic-Geometric Planning

- **Novelty:** Defines a two-way interface in which symbolic and geometric planners interleave decisions and backtrack across levels, rather than following a one-pass task-then-motion pipeline.
- **Limitation / extension:** **Inferred:** effectiveness depends on solver-specific geometric candidates and hand-authored task decomposition/effort levels. **Extension idea:** standardize conflict explanations and learn which layer should branch next from historical search traces.

### T202 — Tower of Hanoi: Representation, Reasoning and Execution

- **Novelty:** Demonstrates an end-to-end ASP representation that combines commonsense/action reasoning with geometric feasibility and physical execution on a canonical long-horizon manipulation puzzle.
- **Limitation / extension:** **Inferred:** a single structured puzzle with known objects provides weak evidence of scalability or open-world robustness. **Extension idea:** use the framework on varied nonmonotone rearrangement suites with noisy perception, learned action models, and recovery metrics.

### T203 — Combining HTN Planning and Geometric Task Planning

- **Novelty:** Connects an HTN planner to a geometric task planner that reasons over task-level geometric entities, and interleaves their refinement and backtracking decisions.
- **Limitation / extension:** **Inferred:** HTN methods, candidate grasps/placements, effort levels, and interaction constraints are predefined, so domain portability is limited. **Extension idea:** induce HTN methods and geometric constraints from traces, then add explicit uncertainty and completeness conditions to the interface.

### T204 — Interval-Bound Constraint Propagation for Geometric Backtracking

- **Novelty:** Uses interval bounds to propagate geometric dependencies across a task plan and reject choices before invoking full path planning, enabling nonchronological geometric backtracking.
- **Limitation / extension:** **Inferred:** axis-aligned intervals lose correlation and become weak in high-dimensional or contact-rich problems. **Extension idea:** combine intervals with convex sets or differentiable relaxations and extract minimal task-level nogoods from the strongest failed relaxation.

### T205 — Manipulation with Multiple Action Types

- **Novelty:** DARRT extends sampling-based planning to sequences mixing transit, rigid transfer, pushing, tilting, pulling, and other non-prehensile primitives in the joint robot-object state space.
- **Limitation / extension:** **Inferred:** action propagation functions are manually authored, no general-purpose symbolic goal planner chooses high-level semantics, and transition subspaces remain hard to sample. **Extension idea:** add learned symbolic abstractions and mode-transition samplers, while using feedback policies to make non-prehensile segments robust.

### T206 — ASP for Collaborative Housekeeping Robotics

- **Novelty:** Combines ASP, commonsense knowledge, continuous geometric and temporal reasoning, monitoring, and collaborative recovery for housekeeping plans with unknown movable obstacles or heavy objects.
- **Limitation / extension:** **Inferred:** the encoded action/failure models and ConceptNet-derived knowledge can be incomplete or wrong, and geometric calls remain expensive. **Extension idea:** probabilistically calibrate extracted knowledge, learn outcomes online, and choose information-gathering/collaboration actions before failure.

### T207 — Hierarchical Task and Motion Planning in the Now

- **Novelty:** Introduces hierarchical planning in the now, refining and executing only enough of a task-motion plan to act, with suggesters proposing continuous parameters when they become relevant.
- **Limitation / extension:** Early commitments and imperfect suggesters can make the approach incomplete, and the original formulation assumes relatively structured, known domains. **Extension idea:** keep a bounded set of alternatives, backtrack across executed abstractions where recovery permits, and combine suggesters with belief-aware learned proposals.

### T208 — Causal plus Geometric Reasoning for Robotic Manipulation

- **Novelty:** Integrates ASP-style causal/action reasoning with geometric reasoning and motion planning so task plans are checked and revised using physical feasibility.
- **Limitation / extension:** **Inferred:** repeated low-level checks and manually encoded causal/geometric predicates limit scale and portability. **Extension idea:** cache generalized conflicts, learn predicate models from simulation/experience, and attach proof or confidence metadata to every external result.

### T209 — Combined Task and Motion Planning for Mobile Manipulation

- **Novelty:** Uses hierarchical/angelic abstractions to couple symbolic mobile-manipulation decisions with continuous feasibility while avoiding full grounding of every low-level trajectory choice.
- **Limitation / extension:** **Inferred:** useful hierarchies and abstraction bounds are domain-designed, and experiments predate broad uncertainty/dynamic-scene evaluation. **Extension idea:** learn abstractions and admissible bounds from solution traces, then extend them to durative resources and closed-loop belief states.

### T210 — Sampling-Based Motion and Symbolic Action Planning

- **Novelty:** SMAP interleaves a symbolic action planner with tree-based continuous exploration, using discrete actions and regions to guide sampling while respecting collision and differential constraints.
- **Limitation / extension:** **Inferred:** performance is sensitive to the symbolic abstraction and region mapping, and sparse/narrow transitions can starve the sampler. **Extension idea:** adapt abstraction resolution online and learn action-conditioned sampling distributions with an exploration floor.

### T211 — Semantic Attachments for Domain-Independent Planning Systems

- **Novelty:** Provides a general mechanism for calling external procedures such as geometric reasoners from a domain-independent planner, establishing a key interface pattern later used throughout TAMP.
- **Limitation / extension:** **Inferred:** opaque modules can break planner assumptions through side effects, incompleteness, or returning only one witness; expensive calls dominate runtime. **Extension idea:** require pure typed contracts, multi-result enumeration, provenance, caching, and fair resampling semantics.

### T212 — Integrating Symbolic and Geometric Planning for Mobile Manipulation

- **Novelty:** Applies semantic attachments in an embodied mobile-manipulation system, letting geometric modules participate during symbolic search and continual replanning rather than only post-validate a plan.
- **Limitation / extension:** **Inferred:** the integration relies on domain-specific modules and deterministic geometric answers, with no general uncertainty or completeness guarantee. **Extension idea:** expose probabilistic outcomes and failure explanations, and benchmark interface modules across robots and planners.

### T213 — A Hybrid Approach to Intricate Motion, Manipulation and Task Planning

- **Novelty:** The expanded aSyMov framework jointly searches symbolic states and context-specific PRMs, explicitly handling topology changes when robots grasp/release objects and trading task branching against roadmap exploration.
- **Limitation / extension:** **Inferred:** maintaining roadmaps for many robot-object compositions can explode, and symbolic/geometric correspondences are manually specified for a known 3D world. **Extension idea:** share roadmap experience across compositions, learn predicate mappings, and update them under uncertain perception.

### T214 — Combining Planning and Motion Planning

- **Novelty:** Automatically extracts high-level logical actions from reachability changes in a motion-planning tree, combines them with nonphysical actions, factors the resulting planning problem, and decodes the abstract plan back into motion.
- **Limitation / extension:** **Inferred:** exhaustive action extraction and factoring become difficult in high-dimensional continuous scenes, and the reachability-derived action vocabulary may miss task semantics. **Extension idea:** incrementally induce only goal-relevant actions, attach learned semantic labels, and validate abstractions with counterexamples.

### T215 — A Robot Task Planner that Merges Symbolic and Geometric Reasoning

- **Novelty:** Presents an early aSyMov implementation in which symbolic and geometric constraints are considered at every search step and the planner chooses between task branching and deeper roadmap learning.
- **Limitation / extension:** **Inferred:** the evidence is a prototype on a geometric Tower-of-Hanoi-style domain with complete geometry and manually linked predicates. **Extension idea:** automate symbolic-geometric interface induction, quantify search fairness/completeness, and test dynamic, uncertain multi-object tasks.

### T216 — aSyMov: A Planner That Deals with Intricate Symbolic and Geometric Problems

- **Novelty:** Establishes explicit links between symbolic positions and continuous configuration subsets, propagates task- and environment-dependent constraints, and uses robot composition plus specialized roadmaps for manipulation modes.
- **Limitation / extension:** **Inferred:** PDDL/geometric links and mode roadmaps are domain engineered, while roadmap growth and multi-robot composition scale poorly. **Extension idea:** represent mode interfaces declaratively, reuse samples across related compositions, and learn which geometric constraints should be instantiated next.

## Foundational precursor analysis — outside the strict/core count

### [P001] Task-Dependent, Human-Oriented Grasp and Placement Selection

- **Novelty:** Couples grasp and placement selection through task, environment, visibility, reachability, and human-oriented constraints rather than optimizing grasp stability in isolation.
- **Limitation / extension:** **Inferred:** it does not search a general symbolic long-horizon task model and relies on engineered human criteria. **Extension idea:** use it as a continuous human-aware refinement module inside TAMP, with personalized preference learning and uncertainty.

### [P002] Randomized Multi-Modal Motion Planning for Humanoid Manipulation

- **Novelty:** Samples sequences and transitions among constrained humanoid manipulation modes, showing how a hybrid contact problem can be searched directly in a multimodal configuration space.
- **Limitation / extension:** **Inferred:** modes and transition samplers are supplied, and narrow transition manifolds can dominate runtime. **Extension idea:** let symbolic task heuristics prioritize modes and learn transition proposals while retaining unbiased samples.

### [P003] Multi-Modal Motion Planning in Non-Expansive Spaces

- **Novelty:** Formalizes sampling-based exploration over multiple continuous manifolds connected by lower-dimensional transitions, supplying a core mathematical model for later TAMP.
- **Limitation / extension:** **Inferred:** the mode graph and transition access assumptions are demanding, and there is no general symbolic task semantics. **Extension idea:** discover mode graphs online and couple them to logical abstractions with explicit completeness conditions.

### [P009] Planning Among Movable Obstacles with Artificial Constraints

- **Novelty:** Encodes obstacle-order and reachability structure as artificial constraints, coupling discrete choices about which object to move to continuous navigation paths.
- **Limitation / extension:** **Inferred:** the method is specialized to NAMO geometry and monotonicity/accessibility heuristics, without a general task language or uncertainty. **Extension idea:** learn obstruction dependencies and integrate stochastic manipulation and task goals through a full TAMP model.

### [P004] Manipulation Planning Among Movable Obstacles

- **Novelty:** Makes obstacle relocation a planning decision within motion search, enabling a robot to decide which object to move and where to restore navigability.
- **Limitation / extension:** **Inferred:** the NAMO structure and manipulation model are specialized and can struggle with many interacting/nonmonotone object moves. **Extension idea:** add task-level causal models, learned obstruction dependencies, and multi-object conflict explanations.

### [P005] Navigation Among Movable Obstacles

- **Novelty:** Provides real-time reasoning over navigation actions and object-moving actions, an early instance of discrete physical intervention coupled to path feasibility.
- **Limitation / extension:** **Inferred:** the focus is navigation with simplified movable obstacles rather than general manipulation tasks or uncertainty. **Extension idea:** embed the NAMO subproblem inside broader TAMP and reason over stochastic pushes, object damage, and alternative task goals.

### [P006] Manipulation Planning with Probabilistic Roadmaps

- **Novelty:** Connects transit and transfer roadmaps through grasp/placement configurations, giving a practical sampling substrate for manipulation's discrete mode switches.
- **Limitation / extension:** **Inferred:** the approach assumes known rigid geometry, stable placements, and relatively small object sets, without a general symbolic task planner. **Extension idea:** add task-directed sampling and incremental roadmap conditioning for multiple movable objects.

### [P007] Two Manipulation Planning Algorithms

- **Novelty:** Systematizes search across discrete placements/grasps and continuous transit/transfer motions, helping establish the multimodal structure later formalized as TAMP.
- **Limitation / extension:** **Inferred:** candidate modes and geometry are idealized and scalability is constrained by combinatorial mode connections. **Extension idea:** sample modes lazily, use task heuristics to prioritize transitions, and reuse experience across instances.

### [P008] A Geometrical Approach to Planning Manipulation Tasks

- **Novelty:** One of the earliest explicit decompositions of manipulation into discrete stable-placement/grasp choices linked by continuous collision-free paths.
- **Limitation / extension:** **Inferred:** the model uses discrete placements/grasps and specialized geometric goals, without general symbolic action semantics, uncertainty, or modern high-DOF validation. **Extension idea:** the natural extension is exactly the later TAMP program: couple mode geometry to expressive task logic and closed-loop perception.
