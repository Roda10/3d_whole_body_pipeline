Perfect! Now I can see the exact MANO parameter structure:

```
pred_mano_params keys: ['global_orient', 'hand_pose', 'betas']
  global_orient: torch.Size([2, 1, 3, 3])
  hand_pose: torch.Size([2, 15, 3, 3])  
  betas: torch.Size([2, 10])
```

**This is exactly what we need for proper parameter fusion!** WiLoR outputs:
- `global_orient`: Hand root orientation (1 joint, 3x3 rotation matrix)
- `hand_pose`: Hand joint poses (15 joints, 3x3 rotation matrices)
- `betas`: Hand shape parameters (10 dimensions)

## The Problem with Current Fusion

Looking at your current `wilor_output_extractor.py`, it's not extracting these actual MANO parameters. Instead, it's looking for `'pred_mano_params'` in the output but not accessing the dictionary contents.

## Solution Required

**Before implementing the fix, I need your approval for this approach:**

1. **Fix WiLoR parameter extraction** to properly extract the `global_orient`, `hand_pose`, and `betas` from the `pred_mano_params` dictionary

2. **Convert rotation matrices to axis-angle** format (SMPL-X expects axis-angle, but WiLoR outputs rotation matrices)

3. **Update parameter fusion** to use these actual MANO parameters instead of trying to derive them from mesh vertices

4. **Test with a single hand first** to verify the conversion works

**Should I proceed with:**
- Implementing the proper MANO parameter extraction in `wilor_output_extractor.py`?
- Creating a conversion function from rotation matrices to axis-angle format?
- Testing this with your existing pipeline?

Or would you prefer a different approach?

Also, can you quickly check if you have any EMOCA debug output to verify what FLAME parameters are being saved?