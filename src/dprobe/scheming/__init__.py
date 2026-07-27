"""Forced-choice roleplay scheming: Qwen picks between honest and deceptive.

Deliberately separate from `dprobe.data.scheming`, which is a different design.
Keeping them apart so the two never get confused in a writeup:

  `dprobe.data.scheming`  build_roleplay()  TEACHER-FORCED. Apollo's honest and
                          deceptive completions are each appended as if the model
                          had said them; the model generates nothing. Reads what
                          the model represents while READING a depicted deception.

  `dprobe.scheming`       roleplay_choice   ON-POLICY FORCED CHOICE. The model is
                          shown both completions as (A) and (B) and picks one.
                          Reads what the model DOES.

The second is the behavioural data. The first is a probe-training source.

Nothing is imported eagerly here on purpose. `python -m
dprobe.scheming.roleplay_choice` imports the package first, and if the package
pulls the submodule in, runpy re-executes it and warns that the module was
"found in sys.modules after import of package" -- which in the worst case gives
you two copies of the module state. Import the submodule directly:

    from dprobe.scheming.roleplay_choice import build_choice_prompt
"""
