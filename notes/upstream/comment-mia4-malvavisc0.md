DRAFT — needs the user's go. Reply on MiaAI-Lab/…-Single-DGX-Spark#4 to malvavisc0.

Thanks for the deployment data and especially the quoting finding: if `--compilation-config` arrives with literal single quotes, the compile-mode row in my point 2 is unmeasured on this kit, not "mode 0", and the same applies to the MTP and YaRN overrides. That makes the A/B you offer the first real one; I would run it as `MAX_NUM_BATCHED_TOKENS=4096` + fixed quoting vs the shipped default, two starts each, before touching anything else.

On prefix caching: agreed, the 1,600-token padding trick assumes a correct align block; with the 8-token ring block chosen as the engine block size the hit restores a zeroed Mamba state, and blazux's two-file fix is the prerequisite. I should have said so in point 3.
