"""GB10 fixes for vLLM's vendored flash-linear-attention (from blazux/qwen3.8-Flash-DGX, contributed by
@Saren-Arterius): (1) the shared-memory gate asks for 100 KiB (102400) but sm_121 has 99 KiB (101376), so
every kernel guarded by check_shared_mem() silently takes its small-tile config; (2) fla#953: tl.dot race on
Blackwell with num_warps=4 in chunk_delta_h -> pin num_warps to 2. Target: the vLLM of the running interpreter
(VLLM_FLA_DIR overrides). `off` restores both."""
import os, sys
def _dir():
    if os.environ.get("VLLM_FLA_DIR"): return os.environ["VLLM_FLA_DIR"]
    import vllm; return os.path.join(os.path.dirname(vllm.__file__), "third_party/flash_linear_attention/ops")
D=_dir(); U=os.path.join(D,"utils.py"); C=os.path.join(D,"chunk_delta_h.py")
EDITS=[(U,"    DEFAULT = 102400  # Default\n","    DEFAULT = 101376  # GB10 (sm_121) has 99 KiB per block; FLAGB10\n"),
       (C,"        for num_warps in [2, 4]\n","        for num_warps in [2]  # fla#953 Blackwell tl.dot race; FLAGB10\n")]
off = sys.argv[1:] and sys.argv[1]=="off"
for path,old,new in EDITS:
    s=open(path).read()
    if off:
        if new in s: open(path,"w").write(s.replace(new,old)); print("  restored", os.path.basename(path))
        else: print("  not installed:", os.path.basename(path))
    else:
        if "FLAGB10" in s: print("  already installed:", os.path.basename(path)); continue
        assert s.count(old)==1, f"anchor in {path}"; open(path,"w").write(s.replace(old,new)); print("  INSTALLED", os.path.basename(path))
