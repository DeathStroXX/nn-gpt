## 3. MANDATORY CONTEXT CHECK — `.agent` DIRECTORY

**At the start of EVERY conversation**, before performing any analysis, code changes, or answering questions related to the `meta_evolution` project, you MUST:

1. **Read the `.agent` directory** at:
   ```
   /shared/ssd/home/b-a-singh/Thesis/clone3/nn-gpt/ab/gpt/brute/ga/meta_evolution/.agent/
   ```

2. **Review all documents** found there, including but not limited to:
   - `CRITICAL_RULES_meta_evol_tune_nngpt.md` — Immutable rules and file analysis
   - `SETUP_GUIDE1.md` — Architecture, deployment, and environment setup
   - `RECENT_FRACTALNET_AUDIT_*.md` — Audit logs of past work
   - `logs/` — Detailed logs from previous agent sessions
   - `skills/` — Learned patterns and reusable knowledge

3. **Obey all rules** defined in those documents. Rules in the `.agent` directory carry the same authority as rules in this file.

4. **Check for conflicts**: If a new task contradicts a rule or finding documented in `.agent`, STOP and inform the user before proceeding.

**Purpose:** This directory serves as the project's institutional memory. It prevents repeating past mistakes, re-doing solved work, and violating critical constraints discovered through hard experience.
