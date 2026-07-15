# Scene Design and Problem Formulation: Reasoning Before Planning in MuJoCo

## 1. Problem Formulation — Nailing It Down

### 1.1 The Two Extremes of Search Strategy

You identified the key design tension. Let me formalize both extremes and argue for a hybrid:

**Strategy A: Object-Specific Search**
- For each missing object, reason about where it might be → search that region → repeat for next object
- Problem: if item1 and item2 are both likely in region1, you'd open region1, grab item1, close it, then open it AGAIN for item2
- Wasteful and unnatural

**Strategy B: Region-Optimal Exploration**
- Plan which regions to explore to maximize expected information gain across ALL missing objects
- Open each region at most once, catalog everything inside
- Problem: purely exploration-driven — doesn't leverage object-specific priors

**Strategy C (Recommended): Hybrid — Object-Aware Region-Optimal Search**

The reasoning module should:
1. **Identify all missing objects** (and potential substitutes) from the goal
2. **Build a joint search-value map**: for each region, estimate the probability that opening it resolves ≥1 missing-object requirement
3. **Plan a region-visitation order** that maximizes cumulative expected resolution per unit search cost
4. **Update the search plan live** as each region is inspected (because finding item1 in region1 eliminates the need to search region2 for item1)

### 1.2 Formal Problem Statement

```
Given:
  - Goal g (natural language)
  - Visible scene state V_k (objects currently visible on open surfaces)
  - Set of searchable regions R = {r1, r2, ..., rm} (closed containers)
  - Spatial commonsense knowledge K (FM-derived)

Step 1 — Precondition Analysis:
  Required(g) = set of objects/roles needed to execute g
  Missing(g, V_k) = Required(g) \ V_k
  
  If Missing = ∅ → proceed to standard TAMP planning
  If Missing ≠ ∅ → proceed to Step 2

Step 2 — Substitution Reasoning:
  For each m ∈ Missing:
    Substitutes(m, g) = {s ∈ V_k ∪ PossiblyHidden | CanSubstitute(s, m, g)}
    
  If ∃ s ∈ V_k that substitutes m → resolve immediately, no search needed
  If all missing objects have visible substitutes → proceed to standard TAMP
  Otherwise → Step 3

Step 3 — Search Planning:
  For each unresolved missing object m (or its substitutes):
    For each region r ∈ R:
      P(m ∈ r | K, scene_context) = spatial commonsense prior
      
  Search_Value(r) = Σ_{m ∈ unresolved} P(m ∈ r | K, scene_context)
  Search_Cost(r) = estimated effort to inspect r (open + look + close)
  
  Priority(r) = Search_Value(r) / Search_Cost(r)
  
  Generate search plan: inspect regions in decreasing Priority order

Step 4 — Search Execution with Live Update:
  For each region r in the search plan:
    Execute: open(r), observe contents
    Update V_k with all newly visible objects
    For each unresolved m:
      If m (or substitute) found in r → mark resolved
    If all missing objects resolved → STOP searching
    Otherwise → re-rank remaining regions (some may now be unnecessary)

Step 5 — Main Task Planning:
  With updated V_k (including found objects/substitutes):
    Proceed with standard TAMP pipeline
```

### 1.3 What Makes This Novel vs. Just "Search Then Plan"

The contribution is NOT just "search for stuff then make coffee." It's:

1. **FM-guided precondition gap detection**: The LLM/VLM identifies WHAT is missing from the goal decomposition
2. **Substitution reasoning**: If the mug is missing but a cup is visible, reason about functional equivalence
3. **Commonsense-ranked multi-region search**: Don't search blindly — use spatial priors to order container inspections
4. **Joint search optimization**: One region inspection serves multiple missing-object queries
5. **Live scene-state feedback**: Each inspection updates the search plan, potentially eliminating remaining searches
6. **Seamless handoff to TAMP**: Once search resolves, the standard execution pipeline takes over

---

## 2. MuJoCo Scene Design

### 2.1 Environment Layout — The Kitchen Workstation

All scenes share a common physical layout (one MuJoCo environment with configurable initial states):

```
┌─────────────────────────────────────────────────────────┐
│                    KITCHEN WORKSTATION                    │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ CABINET  │   │ CABINET  │   │ DRAWER   │            │
│  │   (C1)   │   │   (C2)   │   │  (D1)    │            │
│  │  hinged  │   │  hinged  │   │  slide   │            │
│  │   door   │   │   door   │   │   out    │            │
│  └──────────┘   └──────────┘   └──────────┘            │
│                                                          │
│  ════════════════════════════════════════════            │
│  ║              COUNTERTOP                  ║            │
│  ║                                          ║            │
│  ║   [kettle]  [sugar_jar]   [coffee_jar]   ║            │
│  ║                                          ║            │
│  ║          (visible objects vary            ║            │
│  ║           by scene variant)              ║            │
│  ════════════════════════════════════════════            │
│                                                          │
│  ┌──────────┐                   ┌──────────┐            │
│  │  DRAWER  │                   │   BOX    │            │
│  │   (D2)   │                   │   (B1)   │            │
│  │  slide   │                   │  lidded  │            │
│  │   out    │                   │          │            │
│  └──────────┘                   └──────────┘            │
│                                                          │
│              ┌────────────────┐                          │
│              │  SERVING AREA  │                          │
│              │   (table/tray) │                          │
│              └────────────────┘                          │
│                                                          │
│             🤖 FETCH MOBILE MANIPULATOR                  │
└─────────────────────────────────────────────────────────┘
```

**Searchable regions (containers)**:
- **C1**: Upper cabinet with hinged door (kitchen items: mugs, glasses, bowls)
- **C2**: Upper cabinet with hinged door (pantry items: plates, containers, dry goods)
- **D1**: Drawer — slides out (utensils: spoons, forks, knives, stirrers, spatulas)
- **D2**: Drawer — slides out (tools & misc: tongs, can opener, bottle opener, towels)
- **B1**: Countertop box with lid (storage: tea bags, sugar packets, condiments)

**Countertop** (open surface): Always visible. Contains scene-variant-specific objects.

**Serving Area**: Goal destination for prepared items.

**Navigation Floor**: Fetch starts on the floor in front of the workstation,
facing the containers. Its planning-level mobile base exposes forward,
lateral, and yaw joints; the torso, arm, head, and gripper are independently
actuated.

### 2.2 Object Inventory (Superset — Scenes Use Subsets)

| Object | Typical Location (Commonsense) | Category |
|--------|-------------------------------|----------|
| mug | C1 (cabinet) | drinkware |
| cup | C1 (cabinet) | drinkware |
| glass | C1 (cabinet) | drinkware |
| bowl | C1 or C2 (cabinet) | serveware |
| plate | C2 (cabinet) | serveware |
| small_plate | C2 (cabinet) | serveware |
| spoon | D1 (drawer) | utensil |
| fork | D1 (drawer) | utensil |
| knife | D1 (drawer) | utensil |
| stirrer | D1 (drawer) | utensil |
| spatula | D1 or D2 (drawer) | utensil/tool |
| tongs | D2 (drawer) | tool |
| kettle | countertop | appliance |
| coffee_jar | countertop | ingredient |
| sugar_jar | countertop | ingredient |
| milk_carton | C1 or countertop | ingredient |
| tea_box | B1 (box) | ingredient |
| bread | C2 or countertop | food |
| butter | C1 or countertop | food |
| jam_jar | C2 or B1 | food |
| napkin | D2 (drawer) | accessory |

### 2.3 Substitution Relationships (Commonsense)

These are the "functional equivalence" relationships the reasoning module should infer:

| Primary Object | Acceptable Substitutes | Context |
|---------------|----------------------|---------|
| mug | cup, glass (for hot drinks with care) | drinking vessel |
| stirrer | spoon, fork (less ideal) | stirring |
| plate | small_plate, bowl (less ideal) | serving surface |
| knife | spatula (for spreading) | spreading |
| spoon | fork (for eating, not ideal for soup) | eating utensil |
| tongs | spatula, fork (for grabbing) | grabbing/flipping |

---

## 3. Scene Variants — Progressive Difficulty

### Tier 1: Single Missing Object (Foundation Scenes)

---

#### Scene S1 — "Make Coffee: Missing Mug"
**Goal**: "Make a cup of coffee and place it in the serving area"

**Visible on countertop**: kettle, coffee_jar, sugar_jar, spoon

**Hidden in containers**:
- C1: mug, glass
- C2: plate, bowl
- D1: fork, knife, stirrer
- D2: tongs, napkin
- B1: tea_box

**What must happen**:
1. **Gap detection**: "Making coffee requires a mug/cup → none visible"
2. **Search reasoning**: "Mugs are typically in cabinets → C1 is most likely"
3. **Search execution**: Open C1 → find mug → take mug out
4. **Main task**: pour coffee, add sugar, stir, place in serving area

**Evaluation points**:
- Does the system detect the missing mug?
- Does it search C1 first (commonsense prior)?
- Does it find the mug and proceed with the task?

**Metrics**: Search efficiency (# regions inspected before finding), total task time, task success

---

#### Scene S2 — "Make Coffee: Missing Mug, Wrong Cabinet First"
**Goal**: Same as S1

**Visible on countertop**: kettle, coffee_jar, sugar_jar, spoon

**Hidden in containers**:
- C1: glass, bowl ← mug is NOT here (tests wrong-prior recovery)
- C2: mug, plate ← mug is in the less-likely cabinet
- D1: fork, knife, stirrer
- D2: tongs, napkin
- B1: tea_box

**What must happen**:
1. **Gap detection**: same as S1
2. **Search reasoning**: "Mugs typically in C1" → search C1 first
3. **Search execution**: Open C1 → NO mug, but see glass → **substitution reasoning**: "Could use the glass instead?" → if policy says yes, grab glass. OR continue searching.
4. **If continuing**: re-rank remaining regions → C2 next → find mug
5. **Main task**: complete coffee making

**Evaluation points**:
- Does it recover from a wrong first guess?
- Does it consider the glass as a substitute?
- If it uses the glass, is it counted as a valid alternative?

> [!NOTE]
> This scene tests the substitution-vs-continued-search tradeoff. The "correct" behavior depends on the formulation:
> - **Conservative**: Always look for the exact object first, only substitute if exhausted
> - **Pragmatic**: If a good substitute is found, use it and avoid further search
> - **You should test both policies and compare**

---

#### Scene S3 — "Spread Butter on Bread: Missing Knife"
**Goal**: "Spread butter on bread and place it in the serving area"

**Visible on countertop**: bread, butter

**Hidden in containers**:
- C1: mug, glass
- C2: plate, bowl
- D1: spoon, fork, stirrer ← spatula also here
- D2: knife, tongs, napkin ← knife is here, not in D1!
- B1: tea_box

**What must happen**:
1. **Gap detection**: "Spreading butter requires a knife or spreading utensil → none visible"
2. **Search reasoning**: "Knives are typically in the utensil drawer (D1)"
3. **Search execution**: Open D1 → no knife, but find spoon and spatula
4. **Substitution reasoning**: "A spatula can spread butter" → grab spatula. OR continue to D2 for the actual knife.
5. **Main task**: spread butter, serve

**Evaluation points**:
- Does it reason about WHAT tool is needed (not just "knife" literally, but "spreading implement")?
- Does it recognize spatula as a valid substitute?

---

### Tier 2: Multi-Object Search with Optimization

---

#### Scene S4 — "Setup for Coffee: Multiple Missing Items"
**Goal**: "Make a cup of coffee with milk and sugar, stir it, and serve"

**Visible on countertop**: kettle, coffee_jar

**Missing objects**: mug, milk_carton, spoon, sugar_jar (4 missing items)

**Hidden in containers**:
- C1: mug, milk_carton ← mug here
- C2: plate, bowl
- D1: spoon, fork, knife, stirrer ← spoon here
- D2: tongs, napkin
- B1: sugar_jar ← sugar moved into box

**What must happen**:
1. **Gap detection**: "Coffee requires: mug (missing), spoon/stirrer (missing), sugar (missing), milk"
2. **Joint search planning**: 
   - mug: likely in C1 or C2  
   - spoon: likely in D1  
   - sugar_jar: could be in C2 or B1  
   - **Optimal order**: C1 (might have mug) → D1 (spoon) → B1 (sugar)
3. **Live search execution**:
   - Open C1 → find mug + milk_carton → mug resolved! Also found milk for bonus
   - Update: still need spoon + sugar
   - Open D1 → find spoon → spoon resolved!
   - Update: still need sugar
   - Open B1 → find sugar_jar → sugar resolved!
   - All resolved after 3 inspections (optimal — needed minimum 3 since items are in 3 different places)

**Key evaluation**: Compare against naive object-by-object search:
- **Naive**: Open C1 for mug → Open D1 for spoon → Open C2 for sugar (fail) → Open B1 for sugar = 4 inspections
- **Optimal**: With joint planning, if the system knows sugar might be in B1, it can skip C2 = 3 inspections

---

#### Scene S5 — "Setup for Coffee: Co-located Items"
**Goal**: Same as S4

**Visible on countertop**: kettle, coffee_jar

**Missing**: mug, spoon, sugar_jar

**Hidden in containers**:
- C1: mug, sugar_jar ← BOTH mug and sugar are here!
- C2: plate, bowl
- D1: spoon, fork, knife
- D2: tongs
- B1: tea_box

**What must happen**:
1. **Gap detection**: same as S4
2. **Search execution**: Open C1 → find BOTH mug AND sugar_jar → two items resolved in one inspection!
3. **Live update**: only spoon still missing → Open D1 → found
4. **Total**: 2 inspections instead of potentially 3+

**Key evaluation**: Does the system catalog ALL objects when opening a container? Or does it only look for the target object? The efficient system notices the sugar_jar while looking for the mug.

**This is THE scene that tests your "live scene state fed to reasoning module" idea.** The reasoning module should:
- Open C1 looking for the mug
- See: {mug, sugar_jar}
- Realize sugar_jar was also on the missing list → resolve both
- Update search plan: only spoon left → go to D1

---

#### Scene S6 — "Breakfast Setup: 3 Items, Suboptimal vs Optimal"
**Goal**: "Prepare toast with butter and jam, serve with coffee"

**Visible on countertop**: bread, kettle, coffee_jar

**Missing**: mug, knife (for spreading), jam_jar, butter

**Hidden in containers**:
- C1: mug, butter ← mug + butter co-located
- C2: jam_jar, plate ← jam here
- D1: spoon, fork, knife ← knife here
- D2: tongs, spatula, napkin
- B1: tea_box

**What must happen**:
1. **Gap detection**: Need mug, knife, jam, butter
2. **Joint search planning**:
   - mug → likely C1
   - butter → likely C1 or C2
   - jam → likely C2 or B1
   - knife → likely D1
3. **Optimal search order**: C1 → C2 → D1 (3 inspections, each resolves ≥1 item)
4. **Live execution**:
   - Open C1 → find {mug, butter} → 2 resolved!
   - Open C2 → find {jam_jar, plate} → jam resolved! (plate noted but not needed)
   - Open D1 → find {spoon, fork, knife} → knife resolved!
   - All found in 3 inspections

**Naive approach (object-by-object)**: 
- Search for mug → C1 (find mug, miss butter) → Search for knife → D1 → Search for jam → C2 → Search for butter → back to C1 again = 4 inspections (revisits C1!)

**This is the key scene demonstrating the value of region-optimal over object-specific search.**

---

### Tier 3: Substitution-Heavy Scenes

---

#### Scene S7 — "Make Tea: Primary Item Missing, Substitute Available"
**Goal**: "Make a cup of tea and serve it"

**Visible on countertop**: kettle, sugar_jar

**Hidden in containers**:
- C1: glass, cup ← NO mug anywhere in the scene!
- C2: plate, bowl
- D1: spoon, fork
- D2: tongs, napkin
- B1: tea_box ← tea bags here

**What must happen**:
1. **Gap detection**: "Tea requires: drinking vessel (missing), tea bag (missing)"
2. **Search for mug**: C1 → no mug, but find cup and glass
3. **Substitution reasoning**: "A cup works for tea. A glass could work too. Use the cup."
4. **Search for tea**: B1 → find tea_box
5. **Main task**: make tea in cup, serve

**Key evaluation**: 
- Does the system give up when "mug" is literally absent?
- Does it recognize cup as functionally equivalent?
- Does it reason about which substitute is BEST (cup > glass for hot drinks)?

---

#### Scene S8 — "Stir the Soup: No Stirrer, Reasoning About Alternatives"
**Goal**: "Stir the soup and serve it in a bowl in the serving area"

**Visible on countertop**: pot_with_soup

**Missing**: bowl (for serving), stirring utensil

**Hidden in containers**:
- C1: mug, glass
- C2: bowl, plate ← bowl here
- D1: fork, knife ← NO spoon, NO stirrer anywhere!
- D2: tongs, spatula
- B1: napkin

**What must happen**:
1. **Gap detection**: "Stirring requires a spoon/stirrer (missing), serving requires a bowl (missing)"
2. **Search**: C2 → find bowl → bowl resolved. D1 → find fork, knife but NO safe soup stirrer
3. **Substitution after exhaustion**: search all likely places → no spoon/stirrer found
4. **Reasoning**: "A fork is unsuitable for safely stirring hot soup; a spatula from D2 can work."
5. **Decision**: Open D2 and use the spatula
6. **Main task**: stir soup with substitute, serve in bowl

**Key evaluation**: The system must reason through substitution AFTER genuinely failing to find the primary item. This is different from S2/S7 where the substitute was found alongside the search.

---

### Tier 4: Complex Long-Horizon Scenes

---

#### Scene S9 — "Full Breakfast Service"
**Goal**: "Prepare and serve a complete breakfast: coffee with milk, toast with butter and jam, everything on plates in the serving area"

**Visible on countertop**: kettle, coffee_jar, bread

**Missing**: MANY items — mug, plate, small_plate, knife, butter, jam_jar, milk_carton, spoon

**Hidden in containers**:
- C1: mug, milk_carton, glass
- C2: plate, small_plate, jam_jar
- D1: spoon, fork, knife, stirrer
- D2: tongs, spatula, napkin
- B1: butter, tea_box

**Optimal search**: C1 → C2 → D1 → B1 = 4 inspections (D2 never needed!)

**What must happen**:
1. **Gap detection**: Large set of missing items
2. **Joint search planning with priors**:
   - Drinking vessel → C1
   - Dairy → C1 (co-located with mug!)
   - Plates → C2
   - Condiments → C2 or B1
   - Utensils → D1
   - Butter → C1, C2, or B1
3. **Optimal plan**: C1 first (highest joint value: mug + milk), then C2 (plate + jam), then D1 (knife + spoon), then B1 (butter)
4. **Live updates**: At each step, cross-resolve found items
5. **Task execution**: Long-horizon — make coffee, toast bread, spread butter+jam, plate everything, serve

**This is the most realistic and challenging scene. It tests everything:**
- Multi-object gap detection
- Joint search optimization
- Live state updates
- Long-horizon task after search
- Potential substitutions if some items are unexpectedly absent

---

#### Scene S10 — "Serve Snacks for Two People"
**Goal**: "Prepare and serve tea for two people with snacks — each person needs a cup of tea and a plate with biscuits"

**Visible on countertop**: kettle, sugar_jar

**Missing**: 2× drinking vessel, 2× plate, tea bags, biscuits, 2× spoon

**Hidden in containers**:
- C1: mug, cup, glass ← has 2+ drinking vessels but mixed types
- C2: plate, small_plate, bowl ← has plate options
- D1: spoon, spoon, fork, knife ← 2 spoons available
- D2: tongs, napkin
- B1: tea_box, biscuits ← both tea AND biscuits here (high joint search value!)

**Interesting reasoning challenges**:
- Need 2 drinking vessels — mug + cup works (both serve tea)
- Need 2 plates — plate + small_plate works
- B1 has the highest joint search value (tea_box + biscuits = 2 missing items resolved)

---

## 4. Progressive Experiment Plan

### Phase 1: Foundation (Start Here)
| Scene | Focus | # Missing | # Containers | Substitution? |
|-------|-------|-----------|-------------|---------------|
| S1 | Single object, correct prior | 1 | 5 | No |
| S2 | Single object, wrong prior + substitute | 1 | 5 | Yes (optional) |
| S3 | Single object, tool reasoning | 1 | 5 | Yes (required) |

### Phase 2: Multi-Object Optimization
| Scene | Focus | # Missing | # Containers | Substitution? |
|-------|-------|-----------|-------------|---------------|
| S4 | Multi-object, distributed | 3 | 5 | No |
| S5 | Multi-object, co-located | 3 | 5 | No |
| S6 | Multi-object, revisitation test | 4 | 5 | No |

### Phase 3: Substitution Reasoning
| Scene | Focus | # Missing | # Containers | Substitution? |
|-------|-------|-----------|-------------|---------------|
| S7 | Primary absent, substitute exists | 2 | 5 | Yes (required) |
| S8 | Exhaustive search + forced substitute | 2 | 5 | Yes (after exhaustion) |

### Phase 4: Long-Horizon Integration
| Scene | Focus | # Missing | # Containers | Substitution? |
|-------|-------|-----------|-------------|---------------|
| S9 | Full breakfast — everything combined | 7 | 5 | Possible |
| S10 | Multi-person serving — quantity reasoning | 8+ | 5 | Possible |

---

## 5. Metrics to Measure

### Search Efficiency Metrics
| Metric | Description |
|--------|-------------|
| **Search Inspections (SI)** | Number of containers opened before all objects found |
| **Optimal Inspections (OI)** | Minimum possible inspections given ground-truth object placement |
| **Search Efficiency Ratio (SER)** | OI / SI (1.0 = optimal, <1.0 = wasted inspections) |
| **Redundant Inspections (RI)** | Number of containers opened that contained no useful objects |
| **Region Revisits (RV)** | Number of times a container was opened more than once |
| **Co-Resolution Rate (CRR)** | Fraction of missing objects resolved as "bonus" while searching for another |

### Reasoning Quality Metrics
| Metric | Description |
|--------|-------------|
| **Gap Detection Accuracy (GDA)** | Did the system correctly identify all missing objects? |
| **Prior Ranking Quality (PRQ)** | Was the optimal region ranked #1 by the commonsense prior? |
| **Substitution Validity (SV)** | When a substitute was used, was it functionally valid? |
| **Substitution Efficiency (SE)** | Was the substitute chosen to minimize additional search? |

### Task Completion Metrics (from your existing paper)
| Metric | Description |
|--------|-------------|
| **Task Success Rate (TSR)** | Same as ROBUST TAMP |
| **Partial Goal Completion (PGC)** | Same as ROBUST TAMP |
| **Total Episode Time** | Including search + task execution |
| **FM Invocations** | Number of LLM/VLM calls (search reasoning + task planning) |

---

## 6. Baselines to Compare Against

| Baseline | Strategy | Expected Performance |
|----------|----------|---------------------|
| **Random Search** | Open containers in random order until all items found | Worst SER, no spatial reasoning |
| **Object-Specific Sequential** | For each missing object, separately search most-likely container | Moderate SER, potential revisits |
| **Oracle (Upper Bound)** | Knows exactly where everything is, opens minimum containers | Perfect SER = 1.0 |
| **ROBUST TAMP (your paper)** | Plans with visible objects only, fails/replans when objects missing | Cannot complete task if required objects never become visible through interaction |
| **Proposed System** | Joint search optimization with FM reasoning | Should approach Oracle SER |

---

## 7. Addressing Your "Live Scene State" Question

> "it really is sufficient to look at a region once to know all the objects present in it"

This is a crucial design decision. Here's the implementation:

### The Scene State Buffer

```
class SearchState:
    inspected_regions: set     # Regions already opened and cataloged
    contents: dict             # {region_id: [list of objects found]}
    unresolved: list           # Missing objects still not found
    search_plan: list          # Ordered list of regions still to inspect
    
    def inspect_region(self, region_id):
        """Open a container and catalog ALL contents"""
        objects_found = perception.detect_all_objects(region_id)
        self.inspected_regions.add(region_id)
        self.contents[region_id] = objects_found
        
        # Cross-resolve: check if any found object resolves an unresolved need
        for obj in objects_found:
            for missing in self.unresolved:
                if obj == missing or is_substitute(obj, missing):
                    self.unresolved.remove(missing)
                    # Mark obj as the resolution for missing
        
        # Re-rank remaining search plan (some regions may now be unnecessary)
        self.search_plan = rerank(
            remaining_regions = [r for r in self.search_plan if r not in self.inspected_regions],
            unresolved = self.unresolved
        )
```

### The Key Insight: Search As Information Gathering

Each container inspection is an **information-gathering action**. After each one:
- You learn ALL objects in that container (one-shot full observation)
- You NEVER need to re-open it
- You can immediately update your search plan

This is essentially a **sequential decision problem**:
- State: which regions inspected, which objects found, which still missing
- Actions: which region to inspect next
- Reward: number of missing objects resolved
- Terminal: all missing resolved OR all regions exhausted

The FM provides the **prior** (spatial commonsense), and each inspection provides the **observation** that updates the posterior beliefs about remaining regions.

---

## 8. MuJoCo Implementation Notes

### Container Types to Model

| Container | Mechanism | MuJoCo Joint Type | Objects Inside |
|-----------|-----------|-------------------|----------------|
| Cabinet (C1, C2) | Hinged door | `hinge` joint on door body | Objects placed on shelf inside |
| Drawer (D1, D2) | Slide out | `slide` joint on drawer body | Objects placed inside tray |
| Box (B1) | Lift lid | `hinge` or `free` joint on lid | Objects placed inside box |

### Key Implementation Considerations

1. **Visibility model**: Objects inside closed containers should be invisible to the perception system. When a container is opened, objects inside become part of $V_k$.

2. **Object spawning**: For each scene variant, define which objects go in which containers. Use a config file per scene.

3. **Container interaction**: The robot arm needs to be able to:
   - Reach and grasp door handles
   - Pull drawers open
   - Lift lids
   - Then observe inside (camera or segmentation mask update)
   - Then reach inside to grasp objects

4. **Multiple objects per container**: Each container can hold 2-4 objects. When opened, ALL objects become visible.

5. **Serving area**: A designated surface where completed items should be placed.

### Suggested File Structure
```
mujoco_scenes/
├── assets/
│   ├── kitchen_base.xml          # Shared environment (countertop, cabinets, drawers, box)
│   ├── objects/
│   │   ├── mug.xml
│   │   ├── cup.xml
│   │   ├── plate.xml
│   │   ├── spoon.xml
│   │   ├── knife.xml
│   │   ├── ... (all objects)
│   │   └── pot_with_soup.xml
│   └── containers/
│       ├── cabinet.xml           # Cabinet with hinge door
│       ├── drawer.xml            # Sliding drawer
│       └── box.xml               # Lidded box
├── configs/
│   ├── S1_coffee_missing_mug.yaml
│   ├── S2_coffee_wrong_prior.yaml
│   ├── S3_butter_missing_knife.yaml
│   ├── S4_coffee_multi_missing.yaml
│   ├── S5_coffee_colocated.yaml
│   ├── S6_breakfast_revisitation.yaml
│   ├── S7_tea_substitution.yaml
│   ├── S8_soup_forced_substitute.yaml
│   ├── S9_full_breakfast.yaml
│   └── S10_snacks_for_two.yaml
├── reasoning/
│   ├── gap_detector.py           # Identifies missing objects from goal
│   ├── search_planner.py         # Joint search optimization
│   ├── substitution_reasoner.py  # Functional equivalence reasoning
│   └── scene_state_buffer.py     # Live search state tracker
└── evaluation/
    ├── metrics.py                # All metrics defined above
    └── baselines.py              # Random search, object-specific, oracle
```
